# scraper-weather-obs: the weather that ACTUALLY happened at Bradley Airport (BDL), every hour.
#
# Every hour (:05) this function:
#   1. downloads the last few hours of official airport reports (METARs) for BDL from the Iowa State Mesonet
#   2. cleans them exactly like the Lab 5 training notebook (one report per hour, rounded to the hour, UTC,
#      wind in mph, sky cover in percent, rain "T" -> 0.001)
#   3. saves the raw download:  gs://<bucket>/energy/raw/weather_obs/<UTC time of this run>.csv
#   4. adds/updates rows in:    gs://<bucket>/energy/datasets/bdl_weather_obs_master.csv   (one row per UTC hour)
# HTTP entrypoint: scrape_weather_obs_http        Cleaning matches Module4/Lab5_Building_the_Training_Data.ipynb

import io
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
STATION = os.getenv("WEATHER_STATION", "BDL")
HOURS_BACK = int(os.getenv("HOURS_BACK", "6"))       # re-read a few hours so late reports still land
USER_AGENT = os.getenv("USER_AGENT", "OPIM5512-student-scraper (UConn)")

MASTER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_obs_master.csv"
KNOTS_TO_MPH = 1.15078
SKY_PERCENT = {"CLR": 0, "FEW": 19, "SCT": 44, "BKN": 75, "OVC": 100, "VV": 100}
COLUMNS = ["temp_f", "dewpoint_f", "rel_humidity_pct", "feels_like_f", "wind_speed_mph", "wind_gust_mph",
           "wind_dir_deg", "pressure_inhg", "visibility_mi", "cloud_cover_pct", "precip_in",
           "is_rain", "is_snow", "is_fog"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


def download_reports(start_utc: datetime, end_utc: datetime) -> str:
    """Raw CSV of every routine (3) and special (4) report between two UTC times."""
    fields = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "alti", "vsby",
              "skyc1", "skyc2", "skyc3", "skyc4", "wxcodes", "feel"]
    params = [("station", STATION), ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
              ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
              ("year1", start_utc.year), ("month1", start_utc.month), ("day1", start_utc.day),
              ("hour1", start_utc.hour), ("minute1", 0),
              ("year2", end_utc.year), ("month2", end_utc.month), ("day2", end_utc.day),
              ("hour2", end_utc.hour), ("minute2", 59)]
    for f in fields:
        params.append(("data", f))
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    return r.text


def clean_reports(text: str) -> pd.DataFrame:
    """Same steps as the Lab 5 training notebook. Returns one row per UTC hour (index: hour_utc)."""
    raw = pd.read_csv(io.StringIO(text), na_values=["M"], keep_default_na=False)
    if len(raw) == 0:
        return pd.DataFrame(columns=COLUMNS)
    raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
    raw["hour_utc"] = raw["valid"].dt.round("h")
    raw["minutes_off"] = (raw["valid"] - raw["hour_utc"]).abs().dt.total_seconds() / 60
    raw["is_routine"] = raw["valid"].dt.minute == 51
    raw = raw.sort_values(["hour_utc", "is_routine", "minutes_off"], ascending=[True, False, True])
    h = raw.drop_duplicates("hour_utc", keep="first").set_index("hour_utc")

    out = pd.DataFrame(index=h.index)
    out["temp_f"] = pd.to_numeric(h["tmpf"], errors="coerce")
    out["dewpoint_f"] = pd.to_numeric(h["dwpf"], errors="coerce")
    out["rel_humidity_pct"] = pd.to_numeric(h["relh"], errors="coerce")
    out["feels_like_f"] = pd.to_numeric(h["feel"], errors="coerce")
    sknt = pd.to_numeric(h["sknt"], errors="coerce")
    gust = pd.to_numeric(h["gust"], errors="coerce")
    out["wind_speed_mph"] = sknt * KNOTS_TO_MPH
    out["wind_gust_mph"] = gust.fillna(sknt) * KNOTS_TO_MPH
    out["wind_dir_deg"] = pd.to_numeric(h["drct"], errors="coerce")
    out["pressure_inhg"] = pd.to_numeric(h["alti"], errors="coerce")
    out["visibility_mi"] = pd.to_numeric(h["vsby"], errors="coerce")
    layers = pd.DataFrame(index=h.index)
    for layer in ["skyc1", "skyc2", "skyc3", "skyc4"]:          # up to 4 cloud layers: keep the thickest
        layers[layer] = h[layer].astype(str).str.strip().map(SKY_PERCENT)
    out["cloud_cover_pct"] = layers.max(axis=1)
    out["precip_in"] = pd.to_numeric(h["p01i"].astype(str).replace("T", "0.001"), errors="coerce").fillna(0.0)
    codes = h["wxcodes"].fillna("").astype(str)
    out["is_rain"] = codes.str.contains("RA|DZ").astype(float)
    out["is_snow"] = codes.str.contains("SN|PL|GR|GS").astype(float)
    out["is_fog"] = codes.str.contains("FG|BR").astype(float)
    out.index.name = "hour_utc"
    return out[COLUMNS]


def run_once(dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=HOURS_BACK)
    text = download_reports(start, now)
    new = clean_reports(text)
    new["scraped_at_utc"] = now.isoformat()

    if dry_run:
        return {"status": "ok", "dry_run": True, "hours": len(new), "rows": json.loads(new.reset_index().to_json(orient="records", date_format="iso"))[-3:]}

    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    bucket.blob(f"{ENERGY_PREFIX}/raw/weather_obs/{stamp}.csv").upload_from_string(text, content_type="text/csv")

    blob = bucket.blob(MASTER_KEY)
    if blob.exists():
        old = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col="hour_utc")
        old.index = pd.to_datetime(old.index, utc=True)
        both = pd.concat([old, new])
        master = both[~both.index.duplicated(keep="last")].sort_index()   # newest version of each hour wins
    else:
        master = new.sort_index()
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index_label="hour_utc"), content_type="text/csv")

    last = new.index.max().isoformat() if len(new) else None
    return {"status": "ok", "hours_this_run": int(len(new)), "latest_hour_utc": last,
            "master_rows": int(len(master)), "master_key": MASTER_KEY}


def scrape_weather_obs_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
