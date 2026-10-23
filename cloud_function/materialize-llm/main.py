# main.py
# Build a single, ever-growing CSV from all structured JSONL files.
# Reads:  gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl/*.jsonl
# Writes: gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_master.csv  (atomic publish)

import csv
import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")                      # REQUIRED
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured") # e.g., "structured"

storage_client = storage.Client()

# Accept BOTH runIDs:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

# Stable CSV schema for students
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "source_txt",
    "body_type", "drivetrain", "transmission", "fuel_type", "title_status", "condition",
    "seller_type", "num_owners", "accident_history", "needs_repair",
]

def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:  # populate it.prefixes
        pass
    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]           # e.g. run_id=20251026170002
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)

def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    """Yield dict records from .jsonl under .../run_id=<run_id>/jsonl/ (one JSON per file)."""
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        data = blob.download_as_text()
        line = data.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # ensure required keys exist
            rec.setdefault("run_id", run_id)
            yield rec
        except Exception:
            continue

def _run_id_to_dt(rid: str) -> datetime:
    if RUN_ID_ISO_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    # fallback: now
    return datetime.now(timezone.utc)

def _open_gcs_text_writer(bucket: str, key: str):
    """Open a text-mode writer to GCS; close() will finalize the upload."""
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    # Text mode avoids the flush/finalize pitfall of binary+TextIOWrapper
    return blob.open("w")  # newline handled by csv module


def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = {c: rec.get(c, None) for c in columns}
            w.writerow(row)
            n += 1
    return n  # close() finalizes the upload

def materialize_http(request: Request):
    """
    HTTP POST (no body needed).
    Crawls ALL structured run folders, de-dupes by post_id (keep newest run),
    and writes one CSV directly to .../datasets/listings_master.csv.
    Returns JSON with counts and output path.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not run_ids:
            return jsonify({"ok": False, "error": f"no runs found under {STRUCTURED_PREFIX}/"}), 200

        latest_by_post: Dict[str, Dict] = {}
        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid:
                    continue
                prev = latest_by_post.get(pid)
                if (prev is None) or (_run_id_to_dt(rec.get("run_id", rid)) > _run_id_to_dt(prev.get("run_id", ""))):
                    latest_by_post[pid] = rec

        base = f"{STRUCTURED_PREFIX}/datasets"
        final_key = f"{base}/listings_master_llm.csv"
        rows = _write_csv(latest_by_post.values(), final_key)

        return jsonify({
            "ok": True,
            "runs_scanned": len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written": rows,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200
    except Exception as e:
        # Return a JSON error so you don't just see a plain 500
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
