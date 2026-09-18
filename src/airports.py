"""Static reference data for a small set of major European airports.

Source: ICAO/IATA codes and coordinates are public reference data (OurAirports-style),
hardcoded here to keep the project dependency-free for this lookup.
"""

AIRPORTS = {
    "LFPG": {"name": "Paris Charles de Gaulle", "iata": "CDG", "lat": 49.0097, "lon": 2.5479},
    "LFPO": {"name": "Paris Orly", "iata": "ORY", "lat": 48.7233, "lon": 2.3794},
    "LFMN": {"name": "Nice Côte d'Azur", "iata": "NCE", "lat": 43.6584, "lon": 7.2159},
    "LFLL": {"name": "Lyon Saint-Exupéry", "iata": "LYS", "lat": 45.7256, "lon": 5.0811},
    "EDDF": {"name": "Frankfurt am Main", "iata": "FRA", "lat": 50.0379, "lon": 8.5622},
    "EHAM": {"name": "Amsterdam Schiphol", "iata": "AMS", "lat": 52.3086, "lon": 4.7639},
    "EGLL": {"name": "London Heathrow", "iata": "LHR", "lat": 51.4700, "lon": -0.4543},
    "LEMD": {"name": "Madrid Barajas", "iata": "MAD", "lat": 40.4936, "lon": -3.5668},
}


def get_airport(icao: str) -> dict:
    icao = icao.upper()
    if icao not in AIRPORTS:
        raise KeyError(f"Aéroport inconnu : {icao}. Aéroports supportés : {list(AIRPORTS)}")
    return AIRPORTS[icao]
