---
type: helper
topic: nearest_station
last_updated: 2026-01-14
---
# Nearest-station resolution strategy

1) Alias match (city → likely stations)
2) If no match: geocode place → lat/lon → nearest station using station doc metadata
3) Offer top 3 and ask user to confirm.

Dublin special-case:
- Cork/Galway/Limerick/Waterford → Heuston
- Belfast → Connolly
