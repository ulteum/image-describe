# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "flask>=3.0.0",
#   "openai>=1.0.0",
#   "Pillow>=10.0.0",
#   "requests>=2.31.0",
# ]
# ///

import base64
import io
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
import requests as http
from flask import Flask, g, render_template, request, jsonify
from openai import OpenAI
from PIL import Image

ATLAS_BASE = "https://api.atlascloud.ai/api/v1/model"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB — images sent as base64 data URLs add ~33% overhead

# gunicorn doesn't execute the __main__ block, so configure the logger here so
# DEBUG lines reach the container's stdout.
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
app.logger.addHandler(_h)
app.logger.setLevel(logging.DEBUG)
app.logger.propagate = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# ── SQLite persistence ───────────────────────────────────────────────────────
# API keys are stored in plaintext — the security boundary is the same as the
# previous localStorage setup. If this is ever served beyond localhost, the db
# file holds the same secrets the browser did. Keep the file permissioned.

DB_PATH = Path(os.environ.get("IMGDESC_DB", Path(__file__).parent / "data" / "image-describe.db"))

VALID_KINDS = {"openai", "openai-multipart", "atlascloud"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL UNIQUE,
  kind       TEXT NOT NULL,       -- 'openai' | 'openai-multipart' | 'atlascloud'
  base_url   TEXT,                -- nullable; AtlasCloud has a fixed default
  api_key    TEXT NOT NULL,
  config     TEXT,                -- JSON blob for per-kind extras
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS provider_models (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
  capability  TEXT NOT NULL,      -- 'describe' | 'generate' | 'improve'
  name        TEXT NOT NULL,
  last_used   TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(provider_id, capability, name)
);
CREATE INDEX IF NOT EXISTS idx_pm_lookup
  ON provider_models(provider_id, capability, last_used DESC);

CREATE TABLE IF NOT EXISTS prompts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  type            TEXT NOT NULL,
  prompt          TEXT NOT NULL,
  model           TEXT,
  negative_prompt TEXT,
  metadata        TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_prompts_type_created ON prompts(type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prompts_created      ON prompts(created_at DESC);
"""


def db() -> sqlite3.Connection:
    conn = getattr(g, "_db", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g._db = conn
    return conn


@app.teardown_appcontext
def _close_db(_exc):
    conn = getattr(g, "_db", None)
    if conn is not None:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _get_provider(pid) -> dict | None:
    if pid in (None, "", 0): return None
    try: pid_i = int(pid)
    except (TypeError, ValueError): return None
    row = db().execute(
        "SELECT id, name, kind, base_url, api_key, config FROM providers WHERE id = ?",
        (pid_i,)).fetchone()
    if not row: return None
    d = dict(row)
    try: d["config"] = json.loads(d["config"]) if d["config"] else {}
    except Exception: d["config"] = {}
    return d


def _touch_provider_model(pid: int, capability: str, name: str):
    if not (pid and capability and name): return
    c = db()
    c.execute(
        "INSERT INTO provider_models(provider_id, capability, name) VALUES(?, ?, ?) "
        "ON CONFLICT(provider_id, capability, name) DO UPDATE SET last_used = datetime('now')",
        (int(pid), capability, name))
    c.commit()


def _record_prompt(type_: str, prompt: str, model: str = "", negative: str = "", metadata: dict | None = None):
    if not prompt: return
    c = db()
    c.execute(
        "INSERT INTO prompts(type, prompt, model, negative_prompt, metadata) VALUES(?, ?, ?, ?, ?)",
        (type_, prompt, model or None, negative or None, json.dumps(metadata) if metadata else None),
    )
    c.commit()


init_db()

# ── Describe prompt building ─────────────────────────────────────────────────

SECTION_PROMPTS = {
    "count":       "  - How many characters/people are present, and a brief overall scene context",
    "build":       "  - Physical appearance: age (approximate), gender, body type, height/build",
    "face":        "  - Facial features: face shape, eye color, eyebrows, jaw, notable features",
    "hair":        "  - Hair: color, style, length, texture",
    "skin":        "  - Skin tone and complexion",
    "clothing":    "  - Clothing: describe every garment — name, color, style, fit, material if visible",
    "accessories": "  - Accessories: jewelry, glasses, bags, hats, belts, watches, etc.",
    "expression":  "  - Expression and body pose/stance",
    "marks":       "  - Any distinguishing marks: tattoos, scars, piercings, or other unique features",
}

FORMAT_INSTRUCTIONS = {
    "plain":    "Write your response as plain text. No markdown, no asterisks, no pound signs — just clear prose or simple numbered sections.",
    "markdown": "Format your response in markdown: use ## for each character as a heading, and bullet points for their attributes.",
}


def build_prompt(sections: list, output_format: str) -> str:
    lines = [
        "Analyze this image and provide a detailed character-focused description.",
        "",
        "Cover each of the following:",
        "",
    ]
    for key in SECTION_PROMPTS:
        if key in sections:
            lines.append(SECTION_PROMPTS[key])
    lines += [
        "",
        "If there are no people, describe any notable figures, creatures, or subjects instead. Be specific and thorough.",
        "",
        FORMAT_INSTRUCTIONS.get(output_format, FORMAT_INSTRUCTIONS["plain"]),
    ]
    return "\n".join(lines)


# ── Shared helpers ───────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_mime_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


def to_jpeg_bytes(file_storage) -> tuple[bytes, str]:
    """Return image bytes and mime type, keeping JPEGs as-is and converting others to JPEG."""
    raw = file_storage.read()
    ext = file_storage.filename.rsplit(".", 1)[1].lower() if "." in file_storage.filename else ""
    if ext in ("jpg", "jpeg"):
        return raw, "image/jpeg"
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _url_to_b64(url: str) -> str:
    """Fetch an image URL and return it as a base64 string."""
    resp = http.get(url, timeout=60)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


def make_client(api_key: str, base_url: str):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _float(key: str, default: float) -> float:
    try:
        return float(request.form.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(request.form.get(key, default))
    except (TypeError, ValueError):
        return default


def _atlas_node(body: dict) -> dict:
    """Unwrap AtlasCloud's {"code": 200, "data": {...}} envelope if present."""
    if isinstance(body.get("data"), dict):
        return body["data"]
    return body


def _atlas_outputs_to_b64(outputs: list) -> list[str]:
    results = []
    for out in outputs:
        if not isinstance(out, str) or not out:
            continue
        if out.startswith("data:"):
            results.append(out.split(",", 1)[1])
        elif out.startswith(("http://", "https://")):
            results.append(_url_to_b64(out))
        else:
            results.append(out)  # assume bare base64
    return results


def _atlas_submit_and_poll(api_key: str, model: str, payload: dict, timeout_s: int = 240) -> list[str]:
    """Submit to AtlasCloud generateImage, poll prediction until done, return base64 images."""
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"model": model, "enable_base64_output": True, **payload}

    log_body = {k: (f"<{len(v)} items>" if isinstance(v, list) else
                    (v[:40] + f"…[{len(v)} chars]") if isinstance(v, str) and len(v) > 60 else v)
                for k, v in body.items()}
    app.logger.debug("AtlasCloud POST generateImage: %s", log_body)

    r = http.post(f"{ATLAS_BASE}/generateImage", headers=headers, json=body, timeout=60)
    if not r.ok:
        raise RuntimeError(f"AtlasCloud submit {r.status_code}: {r.text[:500]}")
    data = r.json()
    pred_id = _atlas_node(data).get("id") or data.get("id")
    if not pred_id:
        raise RuntimeError(f"AtlasCloud returned no prediction id: {str(data)[:400]}")
    app.logger.debug("AtlasCloud prediction id=%s, polling…", pred_id)

    deadline = time.monotonic() + timeout_s
    delay = 1.5
    last_status = ""
    while time.monotonic() < deadline:
        time.sleep(delay)
        pr = http.get(f"{ATLAS_BASE}/prediction/{pred_id}", headers=headers, timeout=30)
        if not pr.ok:
            raise RuntimeError(f"AtlasCloud poll {pr.status_code}: {pr.text[:500]}")
        node = _atlas_node(pr.json())
        status = (node.get("status") or "").lower()
        if status != last_status:
            app.logger.debug("AtlasCloud status=%s", status)
            last_status = status
        if status in ("completed", "succeeded"):
            return _atlas_outputs_to_b64(node.get("outputs") or [])
        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"AtlasCloud {status}: {str(node)[:400]}")
        delay = min(delay * 1.25, 4.0)
    raise TimeoutError(f"AtlasCloud polling timed out after {timeout_s}s (id={pred_id})")


def _atlas_upload_media(api_key: str, filename: str, raw: bytes, mime: str) -> str:
    """Upload a local file to AtlasCloud and return its temporary URL."""
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename, raw, mime)}
    r = http.post(f"{ATLAS_BASE}/uploadMedia", headers=headers, files=files, timeout=120)
    if not r.ok:
        raise RuntimeError(f"AtlasCloud uploadMedia {r.status_code}: {r.text[:500]}")
    data = r.json()
    url = _atlas_node(data).get("download_url") or data.get("url")
    if not url:
        raise RuntimeError(f"AtlasCloud uploadMedia returned no url: {str(data)[:400]}")
    return url


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/describe", methods=["POST"])
def describe():
    provider = _get_provider(request.form.get("provider_id"))
    if not provider:
        return jsonify({"error": "Provider is required"}), 400
    model = request.form.get("model", "").strip()
    sections = request.form.getlist("sections") or list(SECTION_PROMPTS.keys())
    output_format = request.form.get("output_format", "plain")
    temperature = _float("temperature", 0.4)
    top_p = _float("top_p", 1.0)
    max_tokens = _int("max_tokens", 2048)
    frequency_penalty = _float("frequency_penalty", 0.1)

    if not model:
        return jsonify({"error": "Model name is required"}), 400
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    image_data = file.read()
    mime_type = get_mime_type(file.filename)
    b64_image = base64.b64encode(image_data).decode("utf-8")
    prompt = build_prompt(sections, output_format)

    _touch_provider_model(provider["id"], "describe", model)
    _record_prompt("describe", prompt, model=model,
                   metadata={"provider_id": provider["id"], "provider_name": provider["name"],
                             "sections": sections, "output_format": output_format,
                             "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                             "frequency_penalty": frequency_penalty})

    try:
        client = make_client(provider["api_key"], provider["base_url"] or "")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        return jsonify({"description": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    provider = _get_provider(request.form.get("provider_id"))
    if not provider:
        return jsonify({"error": "Provider is required"}), 400
    api_key       = provider["api_key"]
    endpoint_url  = (provider["base_url"] or "").strip()
    model         = request.form.get("img_model", "").strip()
    prompt        = request.form.get("prompt", "").strip()
    negative_prompt = request.form.get("negative_prompt", "").strip()
    n             = max(1, min(10, _int("n", 1)))
    size          = request.form.get("size", "").strip()
    quality       = request.form.get("quality", "").strip()
    style         = request.form.get("style", "").strip()

    if not model:
        return jsonify({"error": "Image model is required"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if provider["kind"] != "atlascloud" and not endpoint_url:
        return jsonify({"error": "Provider has no base URL configured"}), 400

    full_prompt = prompt
    if negative_prompt:
        full_prompt += f"\n\nAvoid: {negative_prompt}"

    _touch_provider_model(provider["id"], "generate", model)
    _record_prompt("generate", prompt, model=model, negative=negative_prompt,
                   metadata={"provider_id": provider["id"], "provider_name": provider["name"],
                             "provider_kind": provider["kind"],
                             "n": n, "size": size, "quality": quality, "style": style})

    cfg = provider.get("config") or {}
    image_field_name = (cfg.get("image_field_name") or "imageDataUrl").strip() or "imageDataUrl"
    image_encoding   = cfg.get("image_encoding", "dataurl")  # "dataurl" | "base64"
    headers = {"Authorization": f"Bearer {api_key}"}

    def _opt_float(key):
        val = request.form.get(key, "").strip()
        try: return float(val) if val else None
        except ValueError: return None

    strength             = _opt_float("strength")
    guidance_scale       = _opt_float("guidance_scale")
    num_inference_steps  = _int("num_inference_steps", 0) or None
    seed_raw             = request.form.get("seed", "").strip()
    seed                 = int(seed_raw) if seed_raw.lstrip("-").isdigit() else None

    uploaded_files = [f for f in request.files.getlist("images") if f.filename] or (
        [request.files["image"]] if "image" in request.files and request.files["image"].filename else []
    )
    first_image_bytes = None
    first_image_mime = None
    atlas_image_urls: list[str] = []
    image_data_urls: list[str] = []

    for f in uploaded_files:
        if not allowed_file(f.filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        try:
            if provider["kind"] == "atlascloud":
                raw = f.read()
                mime = get_mime_type(f.filename)
                atlas_image_urls.append(_atlas_upload_media(api_key, f.filename, raw, mime))
            else:
                raw, mime = to_jpeg_bytes(f)
                b64 = base64.b64encode(raw).decode("utf-8")
                image_data_urls.append(f"data:{mime};base64,{b64}")
                if first_image_bytes is None:
                    first_image_bytes = raw
                    first_image_mime = mime
        except Exception as e:
            return jsonify({"error": f"Could not process image: {e}"}), 400

    app.logger.debug(
        "/generate kind=%s model=%s files=%d atlas_urls=%d data_urls=%d keys=%s files_keys=%s",
        provider["kind"], model, len(uploaded_files), len(atlas_image_urls), len(image_data_urls),
        list(request.form.keys()), list(request.files.keys()))

    if provider["kind"] == "atlascloud":
        atlas_payload: dict = {"prompt": full_prompt}
        if size:
            atlas_payload["size"] = size.replace("x", "*")
        if atlas_image_urls:
            atlas_payload["images"] = atlas_image_urls
        try:
            images_b64 = _atlas_submit_and_poll(api_key, model, atlas_payload)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        if not images_b64:
            return jsonify({"error": "AtlasCloud returned no images"}), 500
        return jsonify({"images": images_b64})

    if provider["kind"] == "openai-multipart":
        if not first_image_bytes:
            return jsonify({"error": "Multipart mode requires an image to be uploaded"}), 400
        ext = "jpg" if first_image_mime == "image/jpeg" else "png"
        files = {"image": (f"image.{ext}", first_image_bytes, first_image_mime)}
        data  = {"model": model, "prompt": full_prompt, "n": str(n), "response_format": "b64_json"}
        if size:    data["size"]    = size
        if quality: data["quality"] = quality
        if style:   data["style"]   = style
        req_kwargs = {"files": files, "data": data}
    else:
        payload = {"model": model, "prompt": full_prompt, "n": n, "response_format": "b64_json"}
        if size:                   payload["size"]                  = size
        if quality:                payload["quality"]               = quality
        if style:                  payload["style"]                 = style
        if strength is not None:   payload["strength"]              = strength
        if guidance_scale is not None: payload["guidance_scale"]    = guidance_scale
        if num_inference_steps:    payload["num_inference_steps"]   = num_inference_steps
        if seed is not None:       payload["seed"]                  = seed
        if image_data_urls:
            payload["imageDataUrls"] = image_data_urls
        elif first_image_bytes:
            b64 = base64.b64encode(first_image_bytes).decode("utf-8")
            payload[image_field_name] = f"data:{first_image_mime};base64,{b64}" if image_encoding == "dataurl" else b64
        req_kwargs = {"json": payload}

    if "json" in req_kwargs:
        log_payload = {
            k: (v[:40] + f"…[{len(v)} chars]") if isinstance(v, str) and len(v) > 60 else v
            for k, v in req_kwargs["json"].items()
        }
        app.logger.debug("POST %s  body: %s", endpoint_url, log_payload)
    else:
        app.logger.debug("POST %s  (multipart, fields: %s)", endpoint_url, list(req_kwargs.get("data", {}).keys()))

    try:
        resp = http.post(endpoint_url, headers=headers, timeout=120, **req_kwargs)
    except http.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {e}"}), 500

    try:
        _log_body = resp.json()
        for item in _log_body.get("data", []):
            if "b64_json" in item:
                item["b64_json"] = item["b64_json"][:12] + "…"
        app.logger.debug("Response %s: %s", resp.status_code, _log_body)
    except Exception:
        app.logger.debug("Response %s: %s", resp.status_code, resp.text[:500])

    if not resp.ok:
        try:
            body = resp.json()
            msg = (body.get("error") or {}).get("message") or body.get("message") or body.get("detail") or resp.text[:600]
        except Exception:
            msg = resp.text[:600]
        return jsonify({"error": f"API returned {resp.status_code}: {msg}"}), 500

    try:
        body = resp.json()
    except Exception:
        return jsonify({"error": f"Non-JSON response: {resp.text[:400]}"}), 500

    if "data" in body:
        raw = body["data"]
        images = []
        for item in raw:
            if item.get("b64_json"):
                images.append(item["b64_json"])
            elif item.get("url"):
                images.append(_url_to_b64(item["url"]))
    elif "images" in body:
        images = [i if isinstance(i, str) else i.get("b64_json", "") for i in body["images"]]
    elif "image" in body:
        images = [body["image"]]
    else:
        return jsonify({"error": f"Unrecognised response shape. Keys: {list(body.keys())}. Body: {str(body)[:400]}"}), 500

    return jsonify({"images": [i for i in images if i]})


# ── Prompt improver ──────────────────────────────────────────────────────────

IMPROVE_SYSTEM_PROMPT = """You are an expert prompt engineer for text-to-image models.
The user will give you a draft image prompt. Your job is to refine it into a well-structured
prompt that covers, where applicable: Subject, Action/Pose, Style/Aesthetic, Environment/Setting,
Lighting, Composition/Framing, Color palette, Mood, and Quality/technical modifiers.

Workflow:
1. Read the user's draft.
2. Ask ONE concise clarifying question at a time about whichever dimension is most ambiguous
   or most likely to change the result. Stop asking after at most 5 questions, or sooner if
   you already have enough to write a strong prompt.
3. When ready, output the final rewritten prompt wrapped EXACTLY like:
     <final_prompt>...rewritten prompt here...</final_prompt>
   Put nothing else in that turn — no preamble, no commentary, no markdown.

Keep questions short and focused. Don't lecture. Don't repeat what the user already said."""


@app.route("/improve-prompt", methods=["POST"])
def improve_prompt():
    data = request.get_json(silent=True) or {}
    provider = _get_provider(data.get("provider_id"))
    if not provider:
        return jsonify({"error": "Provider is required"}), 400
    model = (data.get("model") or "").strip()
    messages = data.get("messages") or []
    if not model:
        return jsonify({"error": "Model is required"}), 400
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages array is required"}), 400

    safe_messages = []
    for m in messages:
        if not isinstance(m, dict): continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            safe_messages.append({"role": role, "content": content})
    if not safe_messages:
        return jsonify({"error": "messages must contain at least one user/assistant turn"}), 400

    full_messages = [{"role": "system", "content": IMPROVE_SYSTEM_PROMPT}, *safe_messages]
    _touch_provider_model(provider["id"], "improve", model)

    try:
        client = make_client(provider["api_key"], provider["base_url"] or "")
        resp = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.5,
            max_tokens=1024,
        )
        return jsonify({"reply": resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Persisted data CRUD ──────────────────────────────────────────────────────

def _provider_row_to_dict(r) -> dict:
    d = dict(r)
    try: d["config"] = json.loads(d["config"]) if d.get("config") else {}
    except Exception: d["config"] = {}
    return d


@app.route("/api/providers", methods=["GET"])
def api_providers_list():
    rows = db().execute(
        "SELECT id, name, kind, base_url, api_key, config FROM providers ORDER BY name ASC"
    ).fetchall()
    return jsonify([_provider_row_to_dict(r) for r in rows])


@app.route("/api/providers", methods=["POST"])
def api_providers_upsert():
    data = request.get_json(silent=True) or {}
    pid      = data.get("id")
    name     = (data.get("name") or "").strip()
    kind     = (data.get("kind") or "").strip()
    base_url = (data.get("base_url") or "").strip() or None
    api_key  = (data.get("api_key") or "").strip()
    config   = data.get("config")

    if not name:    return jsonify({"error": "name is required"}), 400
    if kind not in VALID_KINDS:
        return jsonify({"error": f"kind must be one of {sorted(VALID_KINDS)}"}), 400
    if not api_key: return jsonify({"error": "api_key is required"}), 400

    if isinstance(config, str):
        cfg_raw = config.strip()
        if cfg_raw:
            try: json.loads(cfg_raw)
            except Exception as e:
                return jsonify({"error": f"config must be valid JSON: {e}"}), 400
        config_json = cfg_raw or None
    elif isinstance(config, dict):
        config_json = json.dumps(config) if config else None
    else:
        config_json = None

    c = db()
    if pid:
        c.execute(
            "UPDATE providers SET name=?, kind=?, base_url=?, api_key=?, config=?, updated_at=datetime('now') "
            "WHERE id=?",
            (name, kind, base_url, api_key, config_json, int(pid)))
        c.commit()
        return jsonify({"id": int(pid)})
    cur = c.execute(
        "INSERT INTO providers(name, kind, base_url, api_key, config) VALUES(?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "kind = excluded.kind, base_url = excluded.base_url, api_key = excluded.api_key, "
        "config = excluded.config, updated_at = datetime('now')",
        (name, kind, base_url, api_key, config_json))
    c.commit()
    row = c.execute("SELECT id FROM providers WHERE name = ?", (name,)).fetchone()
    return jsonify({"id": row["id"] if row else cur.lastrowid})


@app.route("/api/providers/<int:pid>", methods=["DELETE"])
def api_providers_delete(pid: int):
    c = db()
    c.execute("DELETE FROM providers WHERE id = ?", (pid,))
    c.commit()
    return jsonify({"ok": True})


@app.route("/api/providers/<int:pid>/models", methods=["GET"])
def api_provider_models(pid: int):
    capability = (request.args.get("capability") or "").strip()
    params = [pid]
    where = "provider_id = ?"
    if capability:
        where += " AND capability = ?"
        params.append(capability)
    rows = db().execute(
        f"SELECT name, capability FROM provider_models WHERE {where} ORDER BY last_used DESC LIMIT 50",
        params).fetchall()
    return jsonify([{"name": r["name"], "capability": r["capability"]} for r in rows])


@app.route("/api/prompts", methods=["GET"])
def api_prompts_list():
    type_     = request.args.get("type", "").strip()
    q         = request.args.get("q", "").strip()
    before_id = request.args.get("before_id", "").strip()
    try:
        limit = max(1, min(100, int(request.args.get("limit", 30))))
    except ValueError:
        limit = 30

    clauses, params = [], []
    if type_:
        clauses.append("type = ?"); params.append(type_)
    if q:
        clauses.append("(prompt LIKE ? OR model LIKE ?)")
        like = f"%{q}%"; params += [like, like]
    if before_id.isdigit():
        clauses.append("id < ?"); params.append(int(before_id))
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = db().execute(
        f"SELECT id, type, prompt, model, negative_prompt, metadata, created_at "
        f"FROM prompts {where} ORDER BY id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()

    items = []
    for r in rows:
        item = dict(r)
        if item.get("metadata"):
            try: item["metadata"] = json.loads(item["metadata"])
            except Exception: pass
        items.append(item)
    return jsonify({"items": items, "has_more": len(items) == limit})


@app.route("/api/prompts/<int:pid>", methods=["DELETE"])
def api_prompts_delete(pid: int):
    c = db()
    c.execute("DELETE FROM prompts WHERE id = ?", (pid,))
    c.commit()
    return jsonify({"ok": True})


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, debug=debug, port=port)
