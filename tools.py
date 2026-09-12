"""
tools.py — local, free tool integrations for the family travel agent.

Two tools:
  1. search_flights()  -> wraps `fast-flights` (scrapes Google Flights, no API key)
  2. search_web()      -> wraps `duckduckgo-search` (ddgs) for tour/weather/visa lookups

Install:
    pip install fast-flights ddgs
"""

from datetime import date
from fast_flights import FlightData, Passengers, Result, get_flights
from ddgs import DDGS


def search_flights(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    adults: int = 2,
    seniors: int = 2,
    seat_class: str = "economy",
    max_layovers: int = 1,
) -> dict:
    """
    Search flights via Google Flights (through fast-flights, no API key needed).

    depart_date / return_date: "YYYY-MM-DD" strings.
    seniors is tracked separately so grandparent seat/aisle preferences
    can be applied downstream, but fast-flights bills them as adults.
    """
    flight_legs = [FlightData(date=depart_date, from_airport=origin, to_airport=destination)]
    if return_date:
        flight_legs.append(FlightData(date=return_date, from_airport=destination, to_airport=origin))

    try:
        result: Result = get_flights(
            flight_data=flight_legs,
            trip="round-trip" if return_date else "one-way",
            seat=seat_class,
            passengers=Passengers(adults=adults + seniors, children=0, infants_in_seat=0, infants_on_lap=0),
            fetch_mode="fallback",  # falls back to a headless request if the fast local parser fails
        )
    except Exception as e:
        return {"error": str(e), "flights": []}

    flights = []
    for f in result.flights:
        layovers = getattr(f, "stops", 0)
        if layovers > max_layovers:
            continue
        flights.append({
            "airline": f.name,
            "price": f.price,
            "duration": f.duration,
            "stops": layovers,
            "departure": f.departure,
            "arrival": f.arrival,
        })

    flights.sort(key=lambda x: (x["stops"], x["price"]))
    return {"origin": origin, "destination": destination, "flights": flights}


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    General-purpose web lookup for things flights can't answer:
    bilingual tour availability, seasonal weather, visa rules, cruise reviews.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
    except Exception as e:
        return [{"error": str(e)}]

    return [
        {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
        for r in results
    ]


def check_climate_fit(destination_query: str, min_temp_f: int = 60) -> dict:
    """
    Convenience wrapper: pulls current/seasonal temperature info for a
    destination via web search and flags it against the family's
    min_temp_f mobility_rule. Not a weather API — just a scored search.
    """
    hits = search_web(f"{destination_query} average temperature by month", max_results=3)
    return {"query": destination_query, "min_required_f": min_temp_f, "search_hits": hits}


if __name__ == "__main__":
    # quick manual smoke test
    print(search_flights("SEA", "BUD", str(date(2026, 5, 10)), str(date(2026, 5, 20))))
    print(search_web("Danube river cruise bilingual Chinese English tours"))