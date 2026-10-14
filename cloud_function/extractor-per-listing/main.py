# main.py
# Purpose: Convert raw TXT -> one-line JSON records (.jsonl) in GCS.
# Compatible input layouts:
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/*.txt
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/txt/*.txt
# where <RUN> is either 20251026T170002Z or 20251026170002.
# Output:
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<RUN>/jsonl/<post_id>.jsonl

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# -------------------- ENV --------------------
PROJECT_ID         = os.getenv("PROJECT_ID")
BUCKET_NAME        = os.getenv("GCS_BUCKET")                        # REQUIRED
SCRAPES_PREFIX     = os.getenv("SCRAPES_PREFIX", "scrapes")         # input
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")   # output

# Accept BOTH run id styles:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

storage_client = storage.Client()

# -------------------- SIMPLE REGEX EXTRACTORS --------------------
PRICE_RE      = re.compile(r"\$\s?([0-9,]+)")
YEAR_RE       = re.compile(r"\b(19|20)\d{2}\b")
MAKE_MODEL_RE = re.compile(r"^(?:19|20)\d{2}\n([A-Za-z-]+)\s+([A-Za-z0-9-]+)", re.MULTILINE)

MAX_RUNS_PER_CALL = int(os.getenv("MAX_RUNS_PER_CALL", "24"))  # safety cap: scrape runs handled per call

# -------------------- HELPERS --------------------
def _list_run_ids(bucket: str, scrapes_prefix: str) -> list[str]:
    """
    List run folders under gs://bucket/<scrapes_prefix>/ and return normalized run_ids.
    Accept:
      - <scrapes_prefix>/run_id=20251026T170002Z/
      - <scrapes_prefix>/20251026170002/
    """
    it = storage_client.list_blobs(bucket, prefix=f"{scrapes_prefix}/", delimiter="/")
    for _ in it:
        pass  # populate it.prefixes

    run_ids: list[str] = []
    for pref in getattr(it, "prefixes", []):
        # e.g., 'scrapes/run_id=20251026T170002Z/' OR 'scrapes/20251026170002/'
        tail = pref.rstrip("/").split("/")[-1]
        cand = tail.split("run_id=", 1)[1] if tail.startswith("run_id=") else tail
        if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
            run_ids.append(cand)
    return sorted(run_ids)

def _extracted_run_ids(bucket: str, structured_prefix: str) -> set:
    """run_ids that already have output under <structured_prefix>/run_id=<run_id>/"""
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass  # populate it.prefixes
    done = set()
    for pref in getattr(it, "prefixes", []):
        tail = pref.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            done.add(tail.split("run_id=", 1)[1])
    return done

def _txt_objects_for_run(run_id: str) -> list[str]:
    """
    Return .txt object names for a given run_id.
    Tries (in order) and returns the first non-empty list:
      scrapes/run_id=<run_id>/txt/
      scrapes/run_id=<run_id>/
      scrapes/<run_id>/txt/
      scrapes/<run_id>/
    """
    bucket = storage_client.bucket(BUCKET_NAME)
    candidates = [
        f"{SCRAPES_PREFIX}/run_id={run_id}/txt/",
        f"{SCRAPES_PREFIX}/run_id={run_id}/",
        f"{SCRAPES_PREFIX}/{run_id}/txt/",
        f"{SCRAPES_PREFIX}/{run_id}/",
    ]
    for pref in candidates:
        names = [b.name for b in bucket.list_blobs(prefix=pref) if b.name.endswith(".txt")]
        if names:
            return names
    return []

def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)

def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")

def _parse_run_id_as_iso(run_id: str) -> str:
    """Normalize either run_id style to ISO8601 Z (fallback = now UTC)."""
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif RUN_ID_PLAIN_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            raise ValueError("unsupported run_id")
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# -------------------- PARSE A LISTING --------------------
def parse_listing(text: str) -> dict:
    d = {}

    m = PRICE_RE.search(text)
    if m:
        try:
            d["price"] = int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    y = YEAR_RE.search(text)
    if y:
        try:
            d["year"] = int(y.group(0))
        except ValueError:
            pass

    mm = MAKE_MODEL_RE.search(text)
    if mm:
        d["make"] = mm.group(1)
        d["model"] = mm.group(2)

    # mileage variants
    mi = None
    m1 = re.search(r"(?:mileage|odometer)\s*[:\-]?\s*([\d,]+)", text, re.I)
    if m1:
        try: mi = int(m1.group(1).replace(",", ""))
        except ValueError: mi = None
    if mi is None:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*k\s*(?:mi|mile|miles)\b", text, re.I)
        if m2:
            try: mi = int(float(m2.group(1)) * 1000)
            except ValueError: mi = None
    if mi is None:
        m3 = re.search(r"(\d{1,3}(?:[,\d]{3})*)\s*(?:mi|mile|miles)\b", text, re.I)
        if m3:
            try: mi = int(re.sub(r"[^\d]", "", m3.group(1)))
            except ValueError: mi = None
    if mi is not None:
        d["mileage"] = mi

    return d

# -------------------- HTTP ENTRY --------------------
def extract_http(request: Request):
    """
    Reads every not-yet-extracted run (or one requested run) of TXT listings and writes ONE-LINE JSON records to:
      gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<run_id>/jsonl/<post_id>.jsonl
    Request JSON (optional):
      { "run_id": "<...>", "max_files": 0, "overwrite": false }
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id    = body.get("run_id")
    max_files = int(body.get("max_files") or 0)        # 0 = unlimited
    overwrite = bool(body.get("overwrite") or False)

    # Which runs? A requested run_id, or EVERY scrape run that has not been extracted yet
    # (so no batch is ever skipped, even if the scraper ran twice before the extractor).
    if run_id:
        runs_to_do = [run_id]
    else:
        runs = _list_run_ids(BUCKET_NAME, SCRAPES_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {SCRAPES_PREFIX}/"}), 200
        done = _extracted_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        runs_to_do = [r for r in runs if r not in done]
        if overwrite and runs[-1] not in runs_to_do:
            runs_to_do.append(runs[-1])   # overwrite=true also refreshes the newest run (old behavior)
        runs_to_do = runs_to_do[:MAX_RUNS_PER_CALL]

    processed = written = skipped = errors = 0
    bucket = storage_client.bucket(BUCKET_NAME)
    runs_done = []

    for rid in runs_to_do:
        scraped_at_iso = _parse_run_id_as_iso(rid)
        txt_blobs = _txt_objects_for_run(rid)
        if not txt_blobs:
            continue
        if max_files > 0:
            txt_blobs = txt_blobs[:max_files]
        runs_done.append(rid)

        for name in txt_blobs:
            try:
                text = _download_text(name)
                fields = parse_listing(text)

                post_id = os.path.splitext(os.path.basename(name))[0]
                record = {
                    "post_id": post_id,
                    "run_id": rid,
                    "scraped_at": scraped_at_iso,
                    "source_txt": name,
                    **fields,
                }

                out_key = f"{STRUCTURED_PREFIX}/run_id={rid}/jsonl/{post_id}.jsonl"

                if not overwrite and bucket.blob(out_key).exists():
                    skipped += 1
                else:
                    _upload_jsonl_line(out_key, record)
                    written += 1

            except Exception as e:
                errors += 1
                logging.error(f"Failed {name}: {e}\n{traceback.format_exc()}")

            processed += 1

    result = {
        "ok": True,
        "version": "extractor-v4-all-new-runs",
        "run_id": runs_done[-1] if runs_done else None,
        "runs_processed": runs_done,
        "processed_txt": processed,
        "written_jsonl": written,
        "skipped_existing": skipped,
        "errors": errors
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
