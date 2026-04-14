"""
AI-Powered Train Assistant for Ireland
Deployable version for Hugging Face Spaces.
"""

import os
import json
import sqlite3
import difflib
import re
import glob as glob_mod
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Environment and model init
# ---------------------------------------------------------------------------

load_dotenv(override=True)

open_ai_key = os.getenv("OPENAI_API_KEY")
if open_ai_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

open_ai = OpenAI()

MODEL = "gpt-4.1-mini"
CSV_PATH = os.path.join(os.getcwd(), "/Users/yasinemirkutlu/LLMProjectsUdemy/llm_engineering/AI-Powered-Train-Assistant-for-Ireland-main/irish_rail_services_2026_sample.csv")
DB = os.path.join(os.getcwd(), "irish rail services 2026 sample.db")
TABLE = "rail_services"
LangChain_DB = "rail_vector_db"
KNOWLEDGE_BASE_DIR = "knowledge_base"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant for rail travel in Ireland.
Use the provided tool to look up scheduled train services and indicative fares.
Please provide accurate and concise information.
If the user's date/time is missing, you should ask before providing options.
When listing services, show up to 3 options at earliest with departure time, arrival time, duration, and the fare (as a "from" price).
Please state that you are showing 3 options at earliest.
If the user says travel from dublin to belfast consider that the user refers to dublin connolly station. Do not ask for clarification.
If the user says travel from dublin to cork or galway consider that the user refers to dublin heuston station. Do not ask for clarification.
If user says a date that is in the past, inform them that you can only provide information for current or future dates.
If user says a fuzzy date in the future you can calculate the exact date based on the current date.
If you don't know, say so.
{context}
"""


# ---------------------------------------------------------------------------
# Build SQLite DB from CSV if it doesn't exist
# ---------------------------------------------------------------------------

def _build_sqlite_db():
    """Create the SQLite DB from the CSV file if it doesn't exist yet."""
    if os.path.exists(DB):
        print(f"SQLite DB already exists at {DB}")
        return

    print(f"Building SQLite DB from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    expected = {
        "service_date", "operator", "route_code",
        "departure_station", "arrival_station",
        "departure_time", "arrival_time",
        "duration_min", "adult_single_from_eur", "notes"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    with sqlite3.connect(DB) as conn:
        df.to_sql(TABLE, conn, index=False)
        cur = conn.cursor()
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_service_date ON {TABLE}(service_date);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_route_code   ON {TABLE}(route_code);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_stations     ON {TABLE}(departure_station, arrival_station);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_times        ON {TABLE}(departure_time, arrival_time);")
        conn.commit()

    print(f"SQLite DB created: {DB}")


_build_sqlite_db()


# ---------------------------------------------------------------------------
# Train search tool
# ---------------------------------------------------------------------------

STATION_ALIASES = {
    "dublin": ["Dublin Heuston", "Dublin Connolly"],
    "cork": ["Cork (Kent)"],
    "galway": ["Galway (Ceannt)"],
    "limerick": ["Limerick (Colbert)"],
    "waterford": ["Waterford (Plunkett)"],
    "belfast": ["Belfast (Grand Central)"],
}

_STATIONS_CACHE = None
_DATE_BOUNDS_CACHE = None
_LAST_SEARCH_CACHE = {"query": None, "tickets": []}


def _make_ticket_id(row, service_date: str) -> str:
    return f"{service_date}|{row['route_code']}|{row['departure_time']}|{row['arrival_time']}"


def get_inquired_ticket(selection: int = 1, ticket_id: Optional[str] = None) -> str:
    """Return the ticket from the most recent inquiry (by option number or ticket_id)."""
    tickets = _LAST_SEARCH_CACHE.get("tickets") or []
    if not tickets:
        return "No ticket found yet. Please inquire a route first (e.g., 'Dublin to Cork tomorrow after 14:00')."
    if ticket_id:
        for t in tickets:
            if t.get("ticket_id") == ticket_id:
                return json.dumps(t, ensure_ascii=False, indent=2)
        return f"I couldn't find a ticket with ticket_id='{ticket_id}'."
    try:
        idx = int(selection) - 1
    except Exception:
        idx = 0
    if idx < 0 or idx >= len(tickets):
        return f"Selection out of range. Choose between 1 and {len(tickets)}."
    return json.dumps(tickets[idx], ensure_ascii=False, indent=2)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _load_stations():
    global _STATIONS_CACHE
    if _STATIONS_CACHE is not None:
        return _STATIONS_CACHE
    with sqlite3.connect(DB) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT departure_station FROM {TABLE}")
        deps = [r[0] for r in cur.fetchall()]
        cur.execute(f"SELECT DISTINCT arrival_station FROM {TABLE}")
        arrs = [r[0] for r in cur.fetchall()]
    _STATIONS_CACHE = sorted(set(deps + arrs))
    return _STATIONS_CACHE


def _get_date_bounds():
    global _DATE_BOUNDS_CACHE
    if _DATE_BOUNDS_CACHE is not None:
        return _DATE_BOUNDS_CACHE
    with sqlite3.connect(DB) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT MIN(service_date), MAX(service_date) FROM {TABLE}")
        mn, mx = cur.fetchone()
    _DATE_BOUNDS_CACHE = (mn, mx)
    return _DATE_BOUNDS_CACHE


def _parse_service_date(service_date):
    """Parse fuzzy dates. If None, default to today."""
    if not service_date:
        return date.today().isoformat()
    try:
        # Try ISO format
        return datetime.strptime(service_date, "%Y-%m-%d").date().isoformat()
    except (ValueError, TypeError):
        return date.today().isoformat()


def _clamp_date(service_date: str) -> str:
    mn, mx = _get_date_bounds()
    if service_date < mn:
        return mn
    if service_date > mx:
        return mx
    return service_date


def _candidate_stations(user_text: str):
    if not user_text:
        return []
    key = _normalize(user_text)
    stations = _load_stations()
    if key in STATION_ALIASES:
        return STATION_ALIASES[key]
    for st in stations:
        if _normalize(st) == key:
            return [st]
    subs = [st for st in stations if key in _normalize(st)]
    if subs:
        return subs[:5]
    return difflib.get_close_matches(user_text, stations, n=5, cutoff=0.55)


def search_train_services(departure_station: str,
                          arrival_station: str,
                          service_date: str | None = None,
                          depart_after: str | None = None,
                          limit: int = 3) -> str:
    print(
        f"DB TOOL CALLED: search_train_services({departure_station=}, {arrival_station=}, {service_date=}, {depart_after=}, {limit=})",
        flush=True
    )
    service_date = _clamp_date(_parse_service_date(service_date))
    if not depart_after or not str(depart_after).strip():
        depart_after = "00:00"
    depart_after = str(depart_after).strip()
    dep_candidates = _candidate_stations(departure_station)
    arr_candidates = _candidate_stations(arrival_station)
    if not dep_candidates:
        return f"I couldn't match the departure station '{departure_station}'. Try: Dublin Heuston, Dublin Connolly, Cork (Kent), Galway (Ceannt), Limerick (Colbert), Waterford (Plunkett), Belfast (Grand Central)."
    if not arr_candidates:
        return f"I couldn't match the arrival station '{arrival_station}'. Try: Dublin Heuston, Dublin Connolly, Cork (Kent), Galway (Ceannt), Limerick (Colbert), Waterford (Plunkett), Belfast (Grand Central)."
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        for dep in dep_candidates:
            for arr in arr_candidates:
                cur.execute(
                    f"""
                    SELECT service_date, operator, route_code,
                           departure_station, arrival_station,
                           departure_time, arrival_time,
                           duration_min, adult_single_from_eur
                    FROM {TABLE}
                    WHERE service_date = ?
                      AND departure_station = ?
                      AND arrival_station = ?
                      AND departure_time >= ?
                    ORDER BY departure_time
                    LIMIT ?
                    """,
                    (service_date, dep, arr, depart_after, int(limit)),
                )
                rows = cur.fetchall()
                if rows:
                    tickets = []
                    for r in rows:
                        price_val = None if r["adult_single_from_eur"] is None else float(r["adult_single_from_eur"])
                        ticket = {
                            "ticket_id": _make_ticket_id(r, service_date),
                            "service_date": r["service_date"],
                            "operator": r["operator"],
                            "route_code": r["route_code"],
                            "departure_station": r["departure_station"],
                            "arrival_station": r["arrival_station"],
                            "departure_time": r["departure_time"],
                            "arrival_time": r["arrival_time"],
                            "duration_min": int(r["duration_min"]) if r["duration_min"] is not None else None,
                            "adult_single_from_eur": price_val,
                            "currency": "EUR",
                            "booking_url": "https://booking.cf.irishrail.ie/",
                            "type": "adult_single_from_price",
                        }
                        tickets.append(ticket)
                    _LAST_SEARCH_CACHE["query"] = {
                        "departure_station": dep,
                        "arrival_station": arr,
                        "service_date": service_date,
                        "depart_after": depart_after,
                    }
                    _LAST_SEARCH_CACHE["tickets"] = tickets
                    header = f"Next {len(rows)} train(s) {dep} → {arr} on {service_date} (after {depart_after}):"
                    lines = []
                    for i, t in enumerate(tickets, start=1):
                        price = f"€{t['adult_single_from_eur']:.2f}+" if t["adult_single_from_eur"] is not None else "price N/A"
                        lines.append(
                            f"- Option {i}: {t['departure_time']}–{t['arrival_time']} ({t['duration_min']} min), {price} [{t['route_code']}]"
                        )
                    return header + "\n" + "\n".join(lines)
    return f"No direct scheduled services found for {dep_candidates[0]} → {arr_candidates[0]} on {service_date} after {depart_after}."


# ---------------------------------------------------------------------------
# Current date tool
# ---------------------------------------------------------------------------

def get_current_date(timezone="UTC", include_time=True):
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %H:%M:%S")


# ---------------------------------------------------------------------------
# Tool definitions (JSON schemas)
# ---------------------------------------------------------------------------

train_search_function = {
    "name": "search_train_services",
    "description": "Find scheduled train services in Ireland for a given route and date/time, including duration and indicative fare.",
    "parameters": {
        "type": "object",
        "properties": {
            "departure_station": {
                "type": "string",
                "description": "Departure station or city (e.g., 'Dublin Heuston' or 'Dublin')"
            },
            "arrival_station": {
                "type": "string",
                "description": "Arrival station or city (e.g., 'Cork (Kent)' or 'Cork')"
            },
            "service_date": {
                "type": "string",
                "description": "Travel date (supports YYYY-MM-DD, 'tomorrow', 'Thursday', 'next Thursday', etc.)"
            },
            "depart_after": {
                "type": "string",
                "description": "Only show trains departing at or after this time (HH:MM, optional)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of services to return (default 5)"
            }
        },
        "required": ["departure_station", "arrival_station"],
        "additionalProperties": False
    }
}

get_ticket_function = {
    "name": "get_inquired_ticket",
    "description": "Return the ticket from the most recent inquiry (by option number or ticket_id).",
    "parameters": {
        "type": "object",
        "properties": {
            "selection": {
                "type": "integer",
                "description": "1-based option number (1 = first option)"
            },
            "ticket_id": {
                "type": "string",
                "description": "Exact ticket_id (optional)"
            }
        },
        "additionalProperties": False
    }
}

get_current_date_function = {
    "name": "get_current_date",
    "description": "Returns the current date and time including day of week and hour. Use this when you need to know what day/time it is.",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone (e.g., 'UTC', 'America/New_York'). Defaults to UTC."
            },
            "include_time": {
                "type": "boolean",
                "description": "Whether to include the time (hour, minute, second). Defaults to true."
            }
        }
    }
}

tools = [
    {"type": "function", "function": train_search_function},
    {"type": "function", "function": get_ticket_function},
    {"type": "function", "function": get_current_date_function},
]


# ---------------------------------------------------------------------------
# RAG setup — build Chroma vector store from knowledge_base/
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if not os.path.exists(LangChain_DB):
    print(f"Building Chroma vector store from {KNOWLEDGE_BASE_DIR}/")
    folders = glob_mod.glob(f"{KNOWLEDGE_BASE_DIR}/*")
    documents = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Divided into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=LangChain_DB)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
else:
    print(f"Loading existing Chroma vector store from {LangChain_DB}")
    vectorstore = Chroma(persist_directory=LangChain_DB, embedding_function=embeddings)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)


# ---------------------------------------------------------------------------
# Talker (TTS)
# ---------------------------------------------------------------------------

import tempfile

def talker(message, accent="Irish"):
    response = open_ai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="marin",
        input=message,
        instructions=f"Speak in a natural {accent} English accent."
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(response.content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tool call handler
# ---------------------------------------------------------------------------

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "search_train_services":
            arguments = json.loads(tool_call.function.arguments)
            dep = arguments.get("departure_station")
            arr = arguments.get("arrival_station")
            service_date = arguments.get("service_date")
            depart_after = arguments.get("depart_after")
            limit = arguments.get("limit", 5000)

            details = search_train_services(
                departure_station=dep,
                arrival_station=arr,
                service_date=service_date,
                depart_after=depart_after,
                limit=limit,
            )
            responses.append({"role": "tool", "content": details, "tool_call_id": tool_call.id})

        elif tool_call.function.name == "get_inquired_ticket":
            arguments = json.loads(tool_call.function.arguments or "{}")
            selection = arguments.get("selection", 1)
            ticket_id = arguments.get("ticket_id")
            details = get_inquired_ticket(selection=selection, ticket_id=ticket_id)
            responses.append({"role": "tool", "content": details, "tool_call_id": tool_call.id})

        elif tool_call.function.name == "get_current_date":
            arguments = json.loads(tool_call.function.arguments or "{}")
            timezone = arguments.get("timezone", "GMT")
            include_time = arguments.get("include_time", True)
            details = get_current_date(timezone=timezone, include_time=include_time)
            responses.append({"role": "tool", "content": details, "tool_call_id": tool_call.id})

    return responses


# ---------------------------------------------------------------------------
# Chat function (used by Gradio UI)
# ---------------------------------------------------------------------------

def chat(history):
    # Build system prompt with RAG context from the latest user message
    latest_user_message = ""
    for m in reversed(history):
        if m.get("role") == "user":
            latest_user_message = m.get("content", "")
            break

    context = ""
    if latest_user_message:
        docs = retriever.invoke(latest_user_message)
        context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [{"role": "system", "content": system_prompt}] + history
    response = open_ai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = open_ai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    voice = talker(reply)
    return history, voice


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def put_message_in_chatbot(message, history):
    if history is None:
        history = []
    if message is None or not str(message).strip():
        return "", history
    return "", history + [{"role": "user", "content": str(message).strip()}]


CSS = """
#app-title { font-weight: 800; font-size: 1.25rem; }
#app-subtitle { opacity: 0.85; margin-top: 2px; margin-bottom: 4px; }
.gradio-container { max-width: 1600px !important; }
#chat-wrap { border-radius: 14px; margin-top: -40px; }
"""

AUDIO_UNLOCK_HTML = """
<script>
(function() {
  if (window.__audioUnlockedInit) return;
  window.__audioUnlockedInit = true;

  const unlock = async () => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const buffer = ctx.createBuffer(1, 1, 22050);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      src.start(0);
      if (ctx.state === "suspended") await ctx.resume();
    } catch (e) {}
  };

  document.addEventListener("click", unlock, { once: true });
  document.addEventListener("keydown", unlock, { once: true });
})();
</script>
"""

with gr.Blocks(css=CSS, title="☘️ AI-powered Train Assistant for Ireland", fill_width=True) as ui:
    gr.Markdown(
        """
        <div style="text-align: center;font-size:28px"><strong>☘️🚆🇮🇪 AI-powered Train Assistant for Ireland ☘️🚆🇮🇪</strong></div>
        <div style="text-align: center;font-size:20px"> Powered by frontier models, GPT-4.1-mini and GPT-4o-mini-TTS, with RAG for accurate information.</div>
        <div style="text-align: center;font-size:18px">Timetables • Stations • Tickets • Bikes • Accessibility</div>
        """,
        elem_id="header",
    )

    gr.HTML(AUDIO_UNLOCK_HTML)

    with gr.Row():
        chatbot = gr.Chatbot(
            height=360,
            type="messages",
            show_label=False,
            elem_id="chat-wrap",
        )

    with gr.Row():
        audio_output = gr.Audio(
            autoplay=True,
            type="filepath",
            label="🔊 Spoken reply",
        )

    with gr.Row():
        message = gr.Textbox(
            label="💬 Plan your journey",
            placeholder="Type a message and press Enter… (e.g., Dublin to Belfast on 25 Jan after 14:00)",
            lines=1,
            autofocus=True,
        )

    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_output],
    )

ui.queue()
ui.launch()
