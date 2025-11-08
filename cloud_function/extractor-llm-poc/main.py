# main.py
# Purpose: PoC LLM extractor that reads your existing per-listing JSONL records,
# fetches the original TXT, asks an LLM (Vertex AI) to extract fields, and writes
# a sibling "<post_id>_llm.jsonl" to the NEW 'jsonl_llm/' sub-directory.
#
# FINAL FIXES INCLUDED:
# 1. Schema updated to use "type": "string" + "nullable": True (fixes 'list' object error).
# 2. system_instruction removed from GenerationConfig and merged into prompt (fixes TypeError).
# 3. Prompt uses simple string passing (fixes initial AttributeError).

import os
import re
import json
import logging
import traceback
import time
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# ---- REQUIRED VERTEX AI IMPORTS ----
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content
from google.api_core.exceptions import ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded

# -------------------- ENV --------------------
PROJECT_ID         = os.getenv("PROJECT_ID", "")
REGION             = os.getenv("REGION", "us-central1")
BUCKET_NAME        = os.getenv("GCS_BUCKET", "")
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "vertex").lower()
# Using gemini-2.5-flash as default to address potential memory/performance issues
LLM_MODEL          = os.getenv("LLM_MODEL", "gemini-2.5-flash") 
OVERWRITE_DEFAULT  = os.getenv("OVERWRITE", "false").lower() == "true"
MAX_FILES_DEFAULT  = int(os.getenv("MAX_FILES", "0") or 0)

# GCS READ RETRY - Use default transient error logic
READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

# LLM API RETRY PREDICATE
def _if_llm_retryable(exception):
    """Checks if the exception is transient and should trigger a retry."""
    return isinstance(exception, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

# LLM CALL RETRY
LLM_RETRY = gax_retry.Retry(
    predicate=_if_llm_retryable,
    initial=5.0, maximum=30.0, multiplier=2.0, deadline=180.0,
)

storage_client = storage.Client()
_CACHED_MODEL_OBJ = None

# Accept BOTH run id styles:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")


# -------------------- HELPERS --------------------
def _get_vertex_model() -> GenerativeModel:
    """Initializes and returns the cached Vertex AI model object (thread-safe for CF)."""
    global _CACHED_MODEL_OBJ
    if _CACHED_MODEL_OBJ is None:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID environment variable is missing.")
        
        # Initialize client once per container lifecycle
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL_OBJ = GenerativeModel(LLM_MODEL)
        logging.info(f"Initialized Vertex AI model: {LLM_MODEL} in {REGION}")
    return _CACHED_MODEL_OBJ


def _list_structured_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    """
    List 'structured/run_id=*/' directories and return normalized run_ids.
    """
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass

    runs = []
    for pref in getattr(it, "prefixes", []):
        tail = pref.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            cand = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
                runs.append(cand)
    return sorted(runs)


def _normalize_run_id_iso(run_id: str) -> str:
    """
    Normalize run_id to ISO8601 Z string for provenance.
    """
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


def _list_per_listing_jsonl_for_run(bucket: str, run_id: str) -> list[str]:
    """
    Return *input* per-listing JSONL object names for a given run_id
    (assumes inputs are in 'jsonl/').
    """
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    bucket_obj = storage_client.bucket(bucket)
    names = []
    for b in bucket_obj.list_blobs(prefix=prefix):
        if not b.name.endswith(".jsonl"):
            continue
        names.append(b.name)
    return names


def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)


def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")


def _blob_exists(blob_name: str) -> bool:
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).exists()


def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


# -------------------- VERTEX AI CALL --------------------
def _vertex_extract_fields(raw_text: str) -> dict:
    """
    Ask Gemini to return JSON with exactly: price, year, make, model, mileage.
    """
    model = _get_vertex_model()

    # Strict JSON schema - FIX: Using single type string with nullable=True
    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
        },
        "required": ["price", "year", "make", "model", "mileage"],
        "additionalProperties": False,
    }

    # System instruction (will be prepended to the prompt)
    sys_instr = (
        "Extract ONLY the following fields from the input text. "
        "Return a strict JSON object that conforms to the provided schema. "
        "If a value is not present, use null. "
        "Rules: integers for price/year/mileage; price in USD; mileage in miles; "
        "do not infer values not explicitly present; do not add extra keys."
    )

    # FIX: Combine instruction and text into one prompt string (SDK compatibility)
    prompt = f"{sys_instr}\n\nTEXT:\n{raw_text}" 

    gen_cfg = GenerationConfig(
        # FIX: system_instruction removed to fix TypeError 
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # --- LLM CALL WITH RETRY ---
    max_attempts = 3
    resp = None
    for attempt in range(max_attempts):
        try:
            # Pass the single string prompt
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            break
        except Exception as e:
            if not _if_llm_retryable(e) or attempt == max_attempts - 1:
                logging.error(f"Fatal/non-retryable LLM error or max retries reached: {e}")
                raise
            
            sleep_time = LLM_RETRY._calculate_sleep(attempt)
            logging.warning(f"Transient LLM error on attempt {attempt+1}/{max_attempts}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    if resp is None:
        raise RuntimeError("LLM call failed after all retries.")

    parsed = json.loads(resp.text)

    # Normalize fields post-extraction
    parsed["price"] = _safe_int(parsed.get("price"))
    parsed["year"] = _safe_int(parsed.get("year"))
    parsed["mileage"] = _safe_int(parsed.get("mileage"))
    
    def _norm_str(s):
        if s is None: return None
        s = str(s).strip()
        return s if s else None

    parsed["make"] = _norm_str(parsed.get("make"))
    parsed["model"] = _norm_str(parsed.get("model"))

    return parsed


# -------------------- HTTP ENTRY --------------------
def llm_extract_http(request: Request):
    """
    Reads latest (or requested) run's per-listing JSONL inputs and writes LLM outputs.
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
    if not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing PROJECT_ID env"}), 500
    if LLM_PROVIDER != "vertex":
        return jsonify({"ok": False, "error": "PoC supports LLM_PROVIDER='vertex' only"}), 400

    # Body overrides
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id = body.get("run_id")
    max_files = int(body.get("max_files") or MAX_FILES_DEFAULT or 0)
    overwrite = bool(body.get("overwrite")) if "overwrite" in body else OVERWRITE_DEFAULT

    # Pick newest run if not provided
    if not run_id:
        runs = _list_structured_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {STRUCTURED_PREFIX}/"}), 200
        run_id = runs[-1]

    structured_iso = _normalize_run_id_iso(run_id)

    inputs = _list_per_listing_jsonl_for_run(BUCKET_NAME, run_id)
    if not inputs:
        return jsonify({"ok": True, "run_id": run_id, "processed": 0, "written": 0, "skipped": 0, "errors": 0}), 200
    if max_files > 0:
        inputs = inputs[:max_files]

    logging.info(f"Starting LLM extraction for run_id={run_id} ({len(inputs)} files to process)")

    processed = written = skipped = errors = 0

    for in_key in inputs:
        processed += 1
        try:
            # Read the tiny JSON line (single record)
            raw_line = _download_text(in_key).strip()
            if not raw_line:
                raise ValueError("empty input jsonl")
            base_rec = json.loads(raw_line)

            post_id = base_rec.get("post_id")
            if not post_id:
                raise ValueError("missing post_id in input record")

            source_txt_key = base_rec.get("source_txt")
            if not source_txt_key:
                raise ValueError("missing source_txt in input record")

            # Output path: uses 'jsonl_llm/' folder
            out_prefix = in_key.rsplit("/", 2)[0] + "/jsonl_llm"
            out_key = out_prefix + f"/{post_id}_llm.jsonl"

            if not overwrite and _blob_exists(out_key):
                skipped += 1
                continue

            # Fetch the raw listing TXT; send to LLM
            raw_listing = _download_text(source_txt_key)

            parsed = _vertex_extract_fields(raw_listing)

            # Compose final record
            out_record = {
                "post_id": post_id,
                "run_id": base_rec.get("run_id", run_id),
                "scraped_at": base_rec.get("scraped_at", structured_iso),
                "source_txt": source_txt_key,
                "price": parsed.get("price"),
                "year": parsed.get("year"),
                "make": parsed.get("make"),
                "model": parsed.get("model"),
                "mileage": parsed.get("mileage"),
                "llm_provider": "vertex",
                "llm_model": LLM_MODEL,
                "llm_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            _upload_jsonl_line(out_key, out_record)
            written += 1

        except Exception as e:
            errors += 1
            logging.error(f"LLM extraction failed for {in_key}: {e}\n{traceback.format_exc()}")

    result = {
        "ok": True,
        "version": "extractor-llm-poc",
        "run_id": run_id,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
