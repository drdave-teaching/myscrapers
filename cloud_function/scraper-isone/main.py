# scraper-isone: Connecticut electricity load (and ISO New England's own forecast), every hour.
#
# Needs YOUR free ISO Express account, saved as two GitHub REPOSITORY SECRETS on your fork:
#   ISO_NE_USERNAME and ISO_NE_PASSWORD      (no secrets yet? this function politely skips and does nothing)
#
# Every hour (:15) this function asks ISO New England web services (https://webservices.iso-ne.com/api/v1.1) for:
#   1. official metered hourly load, Connecticut zone (location 4004), for the last few days
#      (these numbers arrive about 2 to 3 days late: they wait for revenue-quality meters)
#        -> gs://<bucket>/energy/datasets/ct_load_hourly_master.csv           (hour_utc, load_mw)
#   2. live five-minute Connecticut load, averaged to each complete hour (available within minutes)
#        -> gs://<bucket>/energy/datasets/ct_load_5min_hourly_master.csv      (hour_utc, load_mw_5min_avg, ...)
#   3. ISO New England's own Connecticut load forecast, every new version it publishes
#        -> gs://<bucket>/energy/datasets/isone_ct_forecast_master.csv        (issued_at_utc, target_hour_utc, ...)
#   Raw responses: gs://<bucket>/energy/raw/isone/<UTC time of this run>_<name>.json
# All times are stored in UTC ("hour_utc" = the hour the value BEGINS), so daylight saving can never duplicate a row.
# HTTP entrypoint: scrape_isone_http

import io
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
ISO_USER = os.getenv("ISO_NE_USERNAME", "")
ISO_PASS = os.getenv("ISO_NE_PASSWORD", "")
LOCATION_ID = os.getenv("ISO_LOCATION_ID", "4004")          # .Z.CONNECTICUT
ZONE_NAME = os.getenv("ISO_ZONE_NAME", ".Z.CONNECTICUT")
DAYS_BACK = int(os.getenv("METERED_DAYS_BACK", "5"))       # metered hourly load arrives 2-3 days late: keep re-asking
BASE = "https://webservices.iso-ne.com/api/v1.1"
EASTERN = ZoneInfo("America/New_York")

KEYS = {"metered": f"{ENERGY_PREFIX}/datasets/ct_load_hourly_master.csv",
        "five_min": f"{ENERGY_PREFIX}/datasets/ct_load_5min_hourly_master.csv",
        "forecast": f"{ENERGY_PREFIX}/datasets/isone_ct_forecast_master.csv"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


def get_json(path: str) -> dict:
    r = requests.get(f"{BASE}/{path}", auth=(ISO_USER, ISO_PASS), headers={"Accept": "application/json"}, timeout=90)
    if r.status_code == 401:
        raise RuntimeError("ISO-NE said 401 (wrong username or password): check the ISO_NE_USERNAME / ISO_NE_PASSWORD secrets")
    r.raise_for_status()
    return r.json()


def as_list(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    return list(x)


def metered_hourly(days) -> tuple:
    rows, raw = [], {}
    for day in days:
        js = get_json(f"realtimehourlydemand/day/{day:%Y%m%d}/location/{LOCATION_ID}")
        raw[f"{day:%Y%m%d}"] = js
        for rec in as_list((js.get("HourlyRtDemands") or {}).get("HourlyRtDemand")):
            rows.append({"hour_utc": pd.Timestamp(rec["BeginDate"]).tz_convert("UTC"), "load_mw": rec.get("Load")})
    df = pd.DataFrame(rows, columns=["hour_utc", "load_mw"])
    return df, raw


def five_minute_hourly(days) -> tuple:
    rows, raw = [], {}
    for day in days:
        js = get_json(f"fiveminuteestimatedzonalload/day/{day:%Y%m%d}")
        raw[f"{day:%Y%m%d}"] = {"rows": len(as_list(js.get("isone_web_services", {}).get("five_min_estimated_zonal_loads", {}).get("five_min_estimated_zonal_load")))}
        for rec in as_list(js.get("isone_web_services", {}).get("five_min_estimated_zonal_loads", {}).get("five_min_estimated_zonal_load")):
            if str(rec.get("load_zone_id")) == LOCATION_ID:
                rows.append({"t": pd.Timestamp(rec["interval_begin_date"]).tz_convert("UTC"),
                             "load": rec.get("estimated_load_mw"), "btm": rec.get("estimated_btm_pv_mw")})
    five = pd.DataFrame(rows, columns=["t", "load", "btm"])
    if len(five) == 0:
        return pd.DataFrame(columns=["hour_utc", "load_mw_5min_avg", "btm_pv_mw_avg", "n_intervals"]), raw
    five["hour_utc"] = five["t"].dt.floor("h")
    g = five.groupby("hour_utc").agg(load_mw_5min_avg=("load", "mean"), btm_pv_mw_avg=("btm", "mean"),
                                     n_intervals=("load", "size")).reset_index()
    g = g[g["n_intervals"] == 12]                               # only complete hours (12 five-minute intervals)
    return g, raw


def isone_forecast() -> tuple:
    js = get_json("reliabilityregionloadforecast/current")
    rows = []
    for rec in as_list((js.get("ReliabilityRegionLoadForecasts") or {}).get("ReliabilityRegionLoadForecast")):
        if rec.get("ReliabilityRegion") == ZONE_NAME:
            rows.append({"issued_at_utc": pd.Timestamp(rec["CreationDate"]).tz_convert("UTC"),
                         "target_hour_utc": pd.Timestamp(rec["BeginDate"]).tz_convert("UTC"),
                         "isone_forecast_mw": rec.get("LoadMw")})
    return pd.DataFrame(rows, columns=["issued_at_utc", "target_hour_utc", "isone_forecast_mw"]), js


def upsert(bucket, key, new, keys, date_cols):
    blob = bucket.blob(key)
    if blob.exists():
        old = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
        for c in date_cols:
            old[c] = pd.to_datetime(old[c], utc=True)
        both = pd.concat([old, new], ignore_index=True)
    else:
        both = new
    both = both.drop_duplicates(keys, keep="last").sort_values(keys)
    bucket.blob(key).upload_from_string(both.to_csv(index=False), content_type="text/csv")
    return len(both)


def run_once(dry_run: bool = False) -> dict:
    if not ISO_USER or not ISO_PASS:
        return {"status": "skipped", "message": "No ISO Express account yet: add ISO_NE_USERNAME and ISO_NE_PASSWORD as "
                "repository secrets on your fork, then re-run the 'Deploy scraper-isone' workflow."}
    now = datetime.now(timezone.utc)
    today = now.astimezone(EASTERN).date()
    metered_days = []
    for back in range(DAYS_BACK, -1, -1):
        metered_days.append(today - timedelta(days=back))
    five_days = [today - timedelta(days=1), today]

    metered, raw_m = metered_hourly(metered_days)
    five, raw_f = five_minute_hourly(five_days)
    fcst, raw_fc = isone_forecast()
    now_iso = now.isoformat()
    metered["scraped_at_utc"] = now_iso
    five["scraped_at_utc"] = now_iso

    summary = {"status": "ok", "metered_hours_seen": int(len(metered)),
               "metered_latest_hour_utc": metered["hour_utc"].max().isoformat() if len(metered) else None,
               "five_min_complete_hours": int(len(five)),
               "five_min_latest_hour_utc": five["hour_utc"].max().isoformat() if len(five) else None,
               "isone_forecast_rows": int(len(fcst)),
               "isone_forecast_issued_at_utc": fcst["issued_at_utc"].max().isoformat() if len(fcst) else None}
    if dry_run:
        summary["dry_run"] = True
        return summary

    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    raws = {"metered": raw_m, "five_min_counts": raw_f, "isone_forecast": raw_fc}
    for name in raws:
        bucket.blob(f"{ENERGY_PREFIX}/raw/isone/{stamp}_{name}.json").upload_from_string(json.dumps(raws[name]), content_type="application/json")

    summary["metered_master_rows"] = upsert(bucket, KEYS["metered"], metered, ["hour_utc"], ["hour_utc"])
    summary["five_min_master_rows"] = upsert(bucket, KEYS["five_min"], five, ["hour_utc"], ["hour_utc"])
    summary["forecast_master_rows"] = upsert(bucket, KEYS["forecast"], fcst, ["issued_at_utc", "target_hour_utc"],
                                             ["issued_at_utc", "target_hour_utc"])
    return summary


def scrape_isone_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
