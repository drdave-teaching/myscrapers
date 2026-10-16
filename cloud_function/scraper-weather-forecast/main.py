# scraper-weather-forecast: what the National Weather Service FORECASTS for Bradley Airport (BDL), every hour.
#
# Every hour (:05) this function:
#   1. downloads the hourly forecast for BDL from forecast.weather.gov (the machine-readable version of
#      https://forecast.weather.gov/MapClick.php?lat=41.9389&lon=-72.6832&FcstType=digital )
#   2. keeps THIS hour and the NEXT 24 hours (lead 0 to 24)
#   3. saves the raw download:  gs://<bucket>/energy/raw/weather_forecast/<UTC time of this run>.xml
#   4. appends to:              gs://<bucket>/energy/datasets/bdl_weather_forecast_master.csv
#      one row per (hour we scraped, hour being forecast). After a week: 168 scrape hours x 25 rows.
#      Columns: scraped_hour_utc, issued_at_utc (when NWS made this forecast), target_hour_utc, lead_hours, weather...
# Why keep every snapshot? An honest 24-hour forecast may only use the forecast that existed at the time.
# HTTP entrypoint: scrape_weather_forecast_http

import io
import json
import logging
import os
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
LAT = os.getenv("FORECAST_LAT", "41.9389")          # Bradley International Airport
LON = os.getenv("FORECAST_LON", "-72.6832")
MAX_LEAD = int(os.getenv("MAX_LEAD_HOURS", "24"))
USER_AGENT = os.getenv("USER_AGENT", "OPIM5512-student-scraper (UConn)")

MASTER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
# (DWML tag, its "type" attribute) -> our column name; units: F, %, mph, degrees, inches
FIELDS = {("temperature", "hourly"): "temp_f", ("temperature", "dew point"): "dewpoint_f",
          ("humidity", "relative"): "rel_humidity_pct", ("wind-speed", "sustained"): "wind_speed_mph",
          ("wind-speed", "gust"): "wind_gust_mph", ("direction", "wind"): "wind_dir_deg",
          ("cloud-amount", "total"): "cloud_cover_pct", ("probability-of-precipitation", "floating"): "precip_chance_pct",
          ("hourly-qpf", "floating"): "precip_in"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


def download_forecast() -> str:
    url = f"https://forecast.weather.gov/MapClick.php?lat={LAT}&lon={LON}&FcstType=digitalDWML"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.text


def parse_forecast(xml_text: str) -> tuple:
    """Returns (issued_at_utc, DataFrame indexed by target_hour_utc with one column per weather field)."""
    root = ET.fromstring(xml_text)
    issued = pd.Timestamp(root.find(".//creation-date").text).tz_convert("UTC")
    starts = [pd.Timestamp(t.text).tz_convert("UTC") for t in root.iter("start-valid-time")]
    table = pd.DataFrame(index=pd.DatetimeIndex(starts, name="target_hour_utc"))
    for el in root.iter():
        key = (el.tag, el.get("type"))
        if key in FIELDS:
            values = []
            for v in el.findall("value"):
                values.append(float(v.text) if v.text not in (None, "") else None)
            if len(values) == len(table):
                table[FIELDS[key]] = values
    return issued, table


def run_once(dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    scraped_hour = pd.Timestamp(now).floor("h")
    xml_text = download_forecast()
    issued, table = parse_forecast(xml_text)

    keep = table[(table.index >= scraped_hour) & (table.index <= scraped_hour + pd.Timedelta(hours=MAX_LEAD))].copy()
    keep["lead_hours"] = ((keep.index - scraped_hour) / pd.Timedelta(hours=1)).astype(int)
    keep["scraped_hour_utc"] = scraped_hour
    keep["issued_at_utc"] = issued
    keep = keep.reset_index()
    first = ["scraped_hour_utc", "issued_at_utc", "target_hour_utc", "lead_hours"]
    keep = keep[first + [c for c in keep.columns if c not in first]]

    if dry_run:
        return {"status": "ok", "dry_run": True, "issued_at_utc": issued.isoformat(), "rows": int(len(keep)),
                "first_rows": json.loads(keep.head(3).to_json(orient="records", date_format="iso"))}

    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    bucket.blob(f"{ENERGY_PREFIX}/raw/weather_forecast/{stamp}.xml").upload_from_string(xml_text, content_type="application/xml")

    blob = bucket.blob(MASTER_KEY)
    if blob.exists():
        old = pd.read_csv(io.BytesIO(blob.download_as_bytes()), parse_dates=["scraped_hour_utc", "issued_at_utc", "target_hour_utc"])
        both = pd.concat([old, keep], ignore_index=True)
        master = both.drop_duplicates(["scraped_hour_utc", "target_hour_utc"], keep="last")
    else:
        master = keep
    master = master.sort_values(["scraped_hour_utc", "target_hour_utc"])
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index=False), content_type="text/csv")
    return {"status": "ok", "issued_at_utc": issued.isoformat(), "rows_this_run": int(len(keep)),
            "scrape_hours_in_master": int(master["scraped_hour_utc"].nunique()), "master_key": MASTER_KEY}


def scrape_weather_forecast_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
