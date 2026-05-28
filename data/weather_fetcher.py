"""
weather_fetcher.py — pulls historical and forecast weather for game locations.
Uses Open-Meteo (100% free, no API key needed).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST   = "https://api.open-meteo.com/v1/forecast"

# Venues where weather has minimal effect (domes / consistent climate)
DOME_VENUES = {
    "Mercedes-Benz Stadium", "AT&T Stadium", "Lucas Oil Stadium",
    "Caesars Superdome", "Ford Field", "US Bank Stadium", "Allegiant Stadium",
    "State Farm Stadium", "NRG Stadium", "Tropicana Field",
    "Dome at America's Center", "Simmons Bank Liberty Stadium",  # Memphis
}


def _weather_params(lat, lon, game_date, is_forecast=False):
    date_str = game_date.strftime("%Y-%m-%d")
    common = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation,snowfall",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Chicago",
    }
    if is_forecast:
        common["start_date"] = date_str
        common["end_date"] = date_str
    else:
        common["start_date"] = date_str
        common["end_date"] = date_str
    return common


def get_game_weather(lat, lon, game_datetime, venue_name=""):
    """
    Return a dict with weather conditions at kickoff:
        temp_f, wind_mph, precip_inch, snowfall_inch, is_dome
    Returns None if the venue is indoors.
    """
    if venue_name in DOME_VENUES:
        return {"temp_f": 72, "wind_mph": 0, "precip_inch": 0,
                "snowfall_inch": 0, "is_dome": True}

    if not lat or not lon:
        return None

    now = datetime.utcnow()
    is_forecast = game_datetime > now + timedelta(days=1)
    url = OPEN_METEO_FORECAST if is_forecast else OPEN_METEO_HISTORICAL

    try:
        params = _weather_params(lat, lon, game_datetime, is_forecast)
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])

        # Find the hour closest to kickoff
        target = game_datetime.strftime("%Y-%m-%dT%H:00")
        if target in times:
            idx = times.index(target)
        elif times:
            idx = 0
        else:
            return None

        return {
            "temp_f":        hourly["temperature_2m"][idx],
            "wind_mph":      hourly["wind_speed_10m"][idx],
            "precip_inch":   hourly["precipitation"][idx],
            "snowfall_inch": hourly["snowfall"][idx],
            "is_dome":       False,
        }
    except Exception:
        return None


def weather_total_adjustment(weather):
    """
    Returns a point adjustment to the total based on weather conditions.
    Negative means the total should be lower (bad weather suppresses scoring).
    """
    if weather is None or weather.get("is_dome"):
        return 0.0

    adj = 0.0
    wind   = weather.get("wind_mph", 0)
    temp   = weather.get("temp_f", 70)
    precip = weather.get("precip_inch", 0)
    snow   = weather.get("snowfall_inch", 0)

    # Wind: each 5 mph above 15 reduces total by ~0.5 pts
    if wind > 15:
        adj -= ((wind - 15) / 5) * 0.5

    # Cold: below 40°F reduces scoring
    if temp < 40:
        adj -= (40 - temp) / 10 * 0.75

    # Rain / snow
    if precip > 0.1:
        adj -= 1.0
    if snow > 0.1:
        adj -= 1.5

    return round(adj, 1)


def weather_spread_adjustment(weather):
    """
    Returns a point adjustment to the spread based on weather.
    (Generally small — weather affects totals more than spreads,
    but passing teams are more hurt than run-heavy teams.)
    Positive = favors home team, negative = favors away team.
    """
    # For now we treat this as symmetric (both teams equally affected).
    # You can add team-specific passing/run rate adjustments later.
    return 0.0


def add_weather_to_schedule(schedule_df, venues_df):
    """
    Given a schedule DataFrame with venue info and a venues DataFrame
    (from CFBD /venues), attach weather conditions to each game.
    Returns the schedule with weather columns appended.
    """
    if venues_df.empty:
        schedule_df["weather"] = None
        return schedule_df

    venue_coords = venues_df[["name", "location.x", "location.y"]].copy()
    venue_coords.columns = ["venue", "lon", "lat"]

    results = []
    for _, row in schedule_df.iterrows():
        venue_name = row.get("venue", "")
        coords = venue_coords[venue_coords["venue"] == venue_name]

        lat = coords["lat"].values[0] if not coords.empty else None
        lon = coords["lon"].values[0] if not coords.empty else None

        game_dt_str = row.get("startDate", "")
        try:
            game_dt = datetime.fromisoformat(game_dt_str.replace("Z", "+00:00"))
        except Exception:
            game_dt = datetime.now()

        weather = get_game_weather(lat, lon, game_dt, venue_name)
        total_adj = weather_total_adjustment(weather)
        spread_adj = weather_spread_adjustment(weather)

        results.append({
            "id": row.get("id"),
            "weather_temp_f":        weather.get("temp_f") if weather else None,
            "weather_wind_mph":      weather.get("wind_mph") if weather else None,
            "weather_precip_inch":   weather.get("precip_inch") if weather else None,
            "weather_is_dome":       weather.get("is_dome", False) if weather else False,
            "weather_total_adj":     total_adj,
            "weather_spread_adj":    spread_adj,
        })

    weather_df = pd.DataFrame(results)
    return schedule_df.merge(weather_df, on="id", how="left")
