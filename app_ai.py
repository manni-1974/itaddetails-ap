# app_ai.py — Two-photo /analyze_pair flow + model repository + scans viewer + JSON API
# - Frontend: posts to /analyze_pair with fileA + fileB + order_no
# - Backend:
#     /analyze_pair    → reads model from photo A, serial from photo B (robust to failures)
#                        and logs a provisional row into scans.csv.
#     /manual_serial_lookup → OEM lookup by serial (Dell only for now)
#     /confirm_row     → updates/creates row in scans.csv and learns into data/models_db.json
#     /view-scans      → view scans.csv in browser (HTML table, last 200 rows)
#     /view-models-db  → view models_db.json in browser
#     /api/scans       → JSON API: last 200 rows of scans.csv (for phone/desktop sync)

import os, re, io, json, time, base64, traceback, shutil
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pandas as pd
from jinja2 import TemplateNotFound

# ===== Optional OpenCV =====
CV2_OK = False
try:
    import cv2
    import numpy as np
    CV2_OK = True
except Exception:
    CV2_OK = False

# ===== ENV =====
API_BASE  = os.getenv("VISION_API_BASE", "https://api.openai.com/v1").rstrip("/")
API_KEY   = os.getenv("VISION_API_KEY", "")
MODEL     = os.getenv("VISION_MODEL", "gpt-4o-mini")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")

ALLOW_LOOKUPS   = os.getenv("ALLOW_LOOKUPS", "1") in ("1","true","True","yes","YES")
LOOKUP_TIMEOUT  = float(os.getenv("LOOKUP_TIMEOUT", "8.0"))
ENABLE_MODEL_INFER = os.getenv("ENABLE_MODEL_INFER", "1") in ("1","true","True","yes","YES")

DATA_CSV = "scans.csv"
UPLOADS  = "uploads"
DATA_DIR = "data"
MODELS_DB_PATH = os.path.join(DATA_DIR, "models_db.json")

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Regulatory → retail hint
REG_MODEL_TO_RETAIL = {"P131G": "Latitude 7410"}

# Quick hints for model→spec guesses
SPEC_HINTS = {
    "Latitude 7410": {
        "cpu": "Intel Core i5-10210U / i7-10610U (10th Gen, vPro optional)",
        "ram": "8 GB default, up to 32 GB DDR4-2666"
    },
    "Latitude 7410 2-in-1": {
        "cpu": "Intel Core i5-10210U / i7-10610U (10th Gen, vPro optional)",
        "ram": "8 GB default, up to 32 GB DDR4-2666"
    },
    "OptiPlex 7010": {
        "cpu": "Varies by config",
        "ram": "4–16 GB typical"
    },
    "OptiPlex 7010 Micro": {
        "cpu": "Intel 12th Gen (e.g., i5-12500T)",
        "ram": "8–32 GB DDR4"
    }
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

# Schema used by vision prompt
SCHEMA = {
  "manufacturer": "",
  "model_no": "",
  "serial": "",
  "product_no": "",
  "reg_model": "",
  "reg_type": "",
  "dpn": "",
  "input_power": "",
  "notes": "",
  "raw_text": "",
  "retail_model_guess": ""
}

# --------- Helpers: recent log + CSV columns ----------

def recent_log():
    if not os.path.exists(DATA_CSV):
        return "(no scans yet)"
    try:
        df = pd.read_csv(DATA_CSV, dtype=str, on_bad_lines="skip", engine="python")
        cols_preferred = [
            "time","order_no",
            "file_a","file_b",            # new
            "file_model","file_serial",   # legacy
            "file",                       # very old
            "manufacturer","model_no","model","serial",
            "product_no","reg_model","reg_type","dpn",
            "input_power","retail_model_guess"
        ]
        cols = [c for c in cols_preferred if c in df.columns]
        if not cols:
            cols = list(df.columns)
        return df[cols].tail(10).to_string(index=False)
    except Exception as e:
        return f"(could not read CSV: {e})"

def ensure_columns(row):
    # Include both new (file_a/file_b) and legacy (file_model/file_serial/file)
    cols = [
        "time","order_no",
        "file_a","file_b",
        "file_model","file_serial",
        "file",            # for very old rows / compatibility
        "manufacturer","model_no","serial",
        "cpu","ram",
        "product_no","reg_model","reg_type","dpn","input_power",
        "retail_model_guess","notes","raw_json"
    ]
    out = {k: "" for k in cols}
    out.update(row)
    return out, cols

# ---------- Imaging ----------

def load_fix_exif(p):
    img = Image.open(p).convert("RGB")
    return ImageOps.exif_transpose(img)

def pillow_contrast_boost(img):
    w, h = img.size
    if w > 20 and h > 20:
        pad = max(2, int(min(w, h) * 0.01))
        img = img.crop((pad, pad, w - pad, h - pad))
    target_min_side = 1600
    w, h = img.size
    if min(w, h) < target_min_side:
        scale = target_min_side / float(min(w, h))
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(2.0)
    g = g.filter(ImageFilter.UnsharpMask(1.2, 160, 3))
    return g.convert("RGB")

def cv2_clahe_boost(img):
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    pad = max(2, int(min(h, w) * 0.01))
    bgr = bgr[pad:h-pad, pad:w-pad]
    target_min = 1600
    h, w = bgr.shape[:2]
    if min(h, w) < target_min:
        s = target_min / float(min(h, w))
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(out, (0,0), 1.0)
    sharp = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return Image.fromarray(sharp)

def preprocess_for_ocr(img):
    if CV2_OK:
        try:
            return cv2_clahe_boost(img)
        except Exception:
            pass
    return pillow_contrast_boost(img)

def inverted_contrast_variant(img):
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageOps.invert(g)
    g = ImageEnhance.Contrast(g).enhance(1.6)
    g = g.filter(ImageFilter.UnsharpMask(1.0, 140, 3))
    return g.convert("RGB")

def jpeg_data_url(img, max_side=2400, quality=88):
    w, h = img.size
    scale = min(1.0, float(max_side)/max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + b64

# ---------- Regex + parsing ----------

SERIAL_PAT       = re.compile(r"(?:S/N|SN|Service\s*Tag|Serial(?:\s*No\.?)?|Serial\s*Number|CN-)\s*[:#]?\s*([A-Z0-9\-]{5,})", re.I)
MODELNO_PAT      = re.compile(r"(?:Model\s*No\.?|Model)\s*[:#]?\s*([A-Z0-9\-\._]+)", re.I)
DPN_PAT          = re.compile(r"\bDP/?N[:\s]*([A-Z0-9\-]+)\b", re.I)
PN_PAT           = re.compile(r"\b(P/?N|Product\s*No\.?)[:\s]*([A-Z0-9\-\._]+)\b", re.I)
REGMODEL_PAT     = re.compile(r"\bReg(?:ulatory)?\s*Model[:\s]*([A-Z0-9\-]+)\b", re.I)
REGTYPE_PAT      = re.compile(r"\bReg(?:ulatory)?\s*Type(?:\s*No\.?)?[:\s]*([A-Z0-9\-]+)\b", re.I)
INPUT_PAT        = re.compile(r"\b(1[29]\.?\d*V|20V)\s*[-–—]?\s*(\d{1,2}\.\d{1,2}A)\b", re.I)
SERVICE_TAG_PAT  = re.compile(r"\b(?:ST|Service\s*Tag)\s*[:#]?\s*([A-Z0-9]{7})\b", re.I)
FAMILY_MODEL_PAT = re.compile(
    r"\b("
    r"Latitude|OptiPlex|Precision|Vostro|XPS|Inspiron|Alienware|Venue|"
    r"ThinkPad|ThinkBook|IdeaPad|Legion|Yoga|"
    r"EliteBook|ProBook|ZBook|Pavilion|Envy|Spectre|"
    r"Surface|Chromebook"
    r")\s+([A-Z0-9\-]+(?:\s+(?:Micro|Mini|Small\s*Form\s*Factor|SFF|Tower|2[-\s]*in[-\s]*1))?)",
    re.I
)

def parse_from_raw_text(norm: dict):
    raw = (norm.get("raw_text") or "") + " " + (norm.get("notes") or "")
    if not norm.get("model_no"):
        m = MODELNO_PAT.search(raw)
        if m:
            norm["model_no"] = m.group(1).strip()
    if not norm.get("dpn"):
        m = DPN_PAT.search(raw)
        if m:
            norm["dpn"] = m.group(1).strip().upper()
    if not norm.get("product_no"):
        m = PN_PAT.search(raw)
        if m:
            norm["product_no"] = m.group(2).strip().upper()
    if not norm.get("reg_model"):
        m = REGMODEL_PAT.search(raw)
        if m:
            norm["reg_model"] = m.group(1).strip().upper()
    if not norm.get("reg_type"):
        m = REGTYPE_PAT.search(raw)
        if m:
            norm["reg_type"] = m.group(1).strip().upper()
    if not norm.get("input_power"):
        m = INPUT_PAT.search(raw)
        if m:
            norm["input_power"] = f"{m.group(1).upper()} {m.group(2)}"
    if not norm.get("retail_model_guess") and (norm.get("manufacturer","").lower() == "dell"):
        rm = (norm.get("reg_model") or "").upper()
        if rm in REG_MODEL_TO_RETAIL:
            norm["retail_model_guess"] = REG_MODEL_TO_RETAIL[rm]
    return norm

def refine_model_with_family(model_no: str, raw_text: str) -> str:
    fam = FAMILY_MODEL_PAT.search(raw_text or "")
    if not fam:
        return model_no
    brand = fam.group(1).strip()
    val   = fam.group(2).strip()
    val = re.sub(r"\b2\s*[- ]?\s*in\s*[- ]?1\b", "2-in-1", val, flags=re.I)
    cand = f"{brand} {val}"
    if (not model_no) or (len(model_no) <= 5 and " " not in model_no):
        return cand
    if brand.lower() in model_no.lower():
        return model_no
    return cand

# ---------- models_db helpers ----------

def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _load_models_db():
    try:
        if os.path.exists(MODELS_DB_PATH):
            with open(MODELS_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print("models_db load error:", repr(e))
    return []

def _save_models_db(records):
    os.makedirs(os.path.dirname(MODELS_DB_PATH), exist_ok=True)
    with open(MODELS_DB_PATH, "w", encoding="utf-8") as f2:
        json.dump(records, f2, ensure_ascii=False, indent=2)

def db_lookup_specs(_manufacturer_unused: str, model_name: str) -> dict:
    key = _norm(model_name)
    if not key:
        return {}
    for rec in _load_models_db():
        mdl = _norm(rec.get("model_no") or rec.get("model") or "")
        alts = rec.get("alt_names") or []
        if mdl == key or any(_norm(a) == key for a in (alts if isinstance(alts, list) else [])):
            out = {}
            if rec.get("cpu"):
                out["cpu"] = rec["cpu"].strip()
            if rec.get("ram"):
                out["ram"] = rec["ram"].strip()
            return out
    return {}

def learn_model_entry(model_no: str, manufacturer: str = "", cpu: str = "", ram: str = ""):
    mdl = (model_no or "").strip()
    if not mdl:
        return
    key = _norm(mdl)
    records = _load_models_db()
    idx = None
    for i, rec in enumerate(records):
        if _norm(rec.get("model_no") or rec.get("model") or "") == key:
            idx = i
            break
    new_rec = {
        "manufacturer": (manufacturer or "").strip(),
        "model_no": mdl,
        "cpu": (cpu or "").strip(),
        "ram": (ram or "").strip(),
        "alt_names": records[idx].get("alt_names", []) if idx is not None else []
    }
    if idx is None:
        records.append(new_rec)
    else:
        records[idx] = new_rec
    _save_models_db(records)

# ---------- LLM helpers ----------

def ai_guess_specs(model_str: str) -> dict:
    model_str = (model_str or "").strip()
    if not API_KEY or not model_str:
        return {}
    for key, v in SPEC_HINTS.items():
        if key.lower() in model_str.lower():
            return {"cpu": v.get("cpu",""), "ram": v.get("ram","")}
    import requests
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT
    prompt = (
        "You are a hardware spec assistant. Given a laptop/desktop marketing model string, "
        "respond with a short JSON object guessing typical CPU family and default RAM. "
        "If multiple CPU options exist, mention concisely.\n"
        "{ \"cpu\": \"...\", \"ram\": \"...\" }\n"
        f"Model: {model_str}\n"
        "Return ONLY the JSON."
    )
    body = {"model": MODEL, "messages": [{"role":"user","content":prompt}], "temperature": 0.1}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=20)
        if r.status_code >= 400:
            return {}
        jd  = r.json()
        txt = jd["choices"][0]["message"]["content"]
        m   = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {}
        obj = json.loads(m.group(0))
        out = {}
        if isinstance(obj, dict):
            if obj.get("cpu"):
                out["cpu"] = str(obj["cpu"]).strip()
            if obj.get("ram"):
                out["ram"] = str(obj["ram"]).strip()
        return out
    except Exception:
        return {}

# ---------- Vision calls ----------

def _vision_extract(img_path, purpose="model"):
    """
    purpose: 'model' or 'serial' — controls prompt/fields we trust.
    - 'model': manufacturer/model + regulatory fields, but NO serial.
    - 'serial': serial only.
    """
    if not API_KEY:
        raise RuntimeError("VISION_API_KEY is not set.")
    import requests
    raw = load_fix_exif(img_path)
    proc = preprocess_for_ocr(raw)
    inv  = inverted_contrast_variant(proc)
    data_proc = jpeg_data_url(proc)
    data_inv  = jpeg_data_url(inv)
    data_raw  = jpeg_data_url(raw)

    if purpose == "model":
        prompt = (
            "Read the DARK manufacturing/regulatory label or brand imprint.\n"
            "Goal: Identify manufacturer and model/marketing model from the label text. "
            "Collect helpful label fields (P/N, DP/N, Reg Model/Type, input power). "
            "DO NOT extract or infer serial in this step.\n\n"
            "Return STRICT JSON like:\n" + json.dumps(SCHEMA, indent=2) + "\n"
            "Rules:\n"
            "- Leave 'serial' empty.\n"
            "- 'model_no' should be the readable marketing/model text "
            "(e.g., 'Latitude 7410', 'ThinkPad T480').\n"
            "- Provide a brief 'raw_text' dump for regex fallback.\n"
            "Return ONLY the JSON."
        )
    else:  # serial
        prompt = (
            "Read the WHITE service/serial sticker or any clear serial area.\n"
            "Goal: Extract only the device serial/service tag. Look for tokens like "
            "'SN', 'S/N', 'Serial', 'Service Tag', 'ST'.\n"
            "Return STRICT JSON like:\n" + json.dumps(SCHEMA, indent=2) + "\n"
            "Rules:\n"
            "- Fill ONLY the 'serial' field if found; other fields should remain empty strings.\n"
            "- Provide 'raw_text' with what you saw around SN tokens.\n"
            "Return ONLY the JSON."
        )

    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":data_proc}},
                {"type":"image_url","image_url":{"url":data_inv}},
                {"type":"image_url","image_url":{"url":data_raw}},
            ]
        }],
        "temperature": 0.0
    }
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            jd = r.json()
            text = jd["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise ValueError(f"No JSON in vision response: {text[:200]}")
            obj = json.loads(m.group(0))
            norm = {k: (obj.get(k) or "").strip() for k in SCHEMA.keys()}
            norm = parse_from_raw_text(norm)
            if purpose == "model":
                norm["serial"] = ""
                norm["model_no"] = refine_model_with_family(
                    norm.get("model_no",""),
                    norm.get("raw_text","")
                )
            else:
                serial_found = norm.get("serial","")
                norm = {k:"" for k in SCHEMA.keys()}
                norm["serial"] = serial_found
                norm["raw_text"] = obj.get("raw_text","")
            if norm.get("input_power"):
                norm["input_power"] = re.sub(r"\s*[-–—]{1,2}\s*", " ", norm["input_power"]).strip()
            return norm
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (429,) or (code is not None and 500 <= code < 600):
                time.sleep(2 ** attempt)
                continue
            raise

# ---------- OEM helpers ----------

def _is_dell_service_tag(tag: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{7}", (tag or "").strip().upper()))

def _lookup_dell_by_tag(tag: str):
    out = {}
    tag = (tag or "").strip().upper()
    if not _is_dell_service_tag(tag):
        return out
    import requests
    endpoints = [
        f"https://www.dell.com/support/api/asset-en-us/assets/getassetssummary?servicetags={tag}",
        f"https://www.dell.com/support/api/asset-en-us/assetinfo/v1/GetAssetHeader?servicetag={tag}",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=LOOKUP_TIMEOUT, headers={"Accept":"application/json"})
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, dict):
                summaries = data.get("assetSummaries") or data.get("AssetSummaries")
                if isinstance(summaries, list) and summaries:
                    s0 = summaries[0]
                    model = (
                        s0.get("productLineDescription") or
                        s0.get("Model") or
                        s0.get("productId") or
                        ""
                    )
                    if model:
                        out["manufacturer"] = "DELL"
                        out["model_no"] = str(model).strip()
                        out["cpu"] = (s0.get("cpu") or s0.get("cpuDescription") or "").strip()
                        out["ram"] = (s0.get("memory") or s0.get("ram") or "").strip()
                        return out
                hdr = data.get("AssetHeaderData") or data.get("assetHeaderData")
                if isinstance(hdr, dict):
                    model = hdr.get("MachineDescription") or hdr.get("SystemModel") or ""
                    if model:
                        out["manufacturer"] = "DELL"
                        out["model_no"] = str(model).strip()
                        return out
        except Exception:
            continue
    return out

def oem_lookup(serial: str) -> dict:
    sn = (serial or "").strip().upper()
    if not sn:
        return {}
    if _is_dell_service_tag(sn):
        got = _lookup_dell_by_tag(sn)
        if got and (not got.get("cpu") or not got.get("ram")) and got.get("model_no"):
            guess = ai_guess_specs(got["model_no"])
            if guess.get("cpu"):
                got["cpu"] = guess["cpu"]
            if guess.get("ram"):
                got["ram"] = guess["ram"]
        return got
    return {}

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html", log=recent_log())

@app.route("/selftest")
def selftest():
    import requests
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT
    body = {"model": MODEL, "messages": [{"role":"user","content":"Reply 'ok'."}], "temperature": 0.0}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code >= 400:
            return jsonify({"ok": False, "status": r.status_code, "body": r.text}), 500
        txt = r.json()["choices"][0]["message"]["content"]
        return jsonify({"ok": True, "reply": txt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# --- MAIN ENDPOINT YOUR UI CALLS: /analyze_pair ---

@app.route("/analyze_pair", methods=["POST"])
def analyze_pair():
    """
    Expects:
      - fileA: model/label photo
      - fileB: serial sticker photo
      - order_no: optional
    Matches your current index.html JS.
    Always returns JSON; if vision fails, fields may be blank.

    ALSO: logs a provisional row into scans.csv so /view-scans and /api/scans update
    right after Analyze, before Confirm.
    """
    try:
        fA = request.files.get("fileA")
        fB = request.files.get("fileB")
        order_no = (request.form.get("order_no") or "").strip()

        if not fA or not fB:
            return jsonify({"error":"need_both_files"}), 400

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save A
        nameA = f"{ts}_A_{fA.filename or 'a.jpg'}"
        pA = os.path.join(UPLOADS, nameA)
        fA.save(pA)
        try:
            imgA = load_fix_exif(pA); imgA.save(pA, "JPEG", quality=90, optimize=True)
        except Exception:
            pass

        # Save B
        nameB = f"{ts}_B_{fB.filename or 'b.jpg'}"
        pB = os.path.join(UPLOADS, nameB)
        fB.save(pB)
        try:
            imgB = load_fix_exif(pB); imgB.save(pB, "JPEG", quality=90, optimize=True)
        except Exception:
            pass

        # Run vision, but don't crash if it fails
        model_info = {}
        serial_info = {}

        try:
            if API_KEY:
                model_info = _vision_extract(pA, purpose="model")
        except Exception as e:
            print("MODEL VISION ERROR:", repr(e), flush=True)

        try:
            if API_KEY:
                serial_info = _vision_extract(pB, purpose="serial")
        except Exception as e:
            print("SERIAL VISION ERROR:", repr(e), flush=True)

        manufacturer = (model_info.get("manufacturer") or "").strip()
        model_no = (model_info.get("model_no") or model_info.get("retail_model_guess") or "").strip()
        serial = (serial_info.get("serial") or "").strip().upper()

        # CPU / RAM via model DB + guessing
        cpu = ""
        ram = ""
        if model_no:
            hit = db_lookup_specs(manufacturer, model_no)
            cpu = hit.get("cpu","") or cpu
            ram = hit.get("ram","") or ram
            if (not cpu or not ram) and ENABLE_MODEL_INFER:
                guess = ai_guess_specs(model_no)
                if not cpu and guess.get("cpu"):
                    cpu = guess["cpu"]
                if not ram and guess.get("ram"):
                    ram = guess["ram"]

        out = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "order_no": order_no,
            "file_a": nameA,
            "file_b": nameB,
            "file_model": nameA,
            "file_serial": nameB,

            "manufacturer": manufacturer,
            "model_no": model_no,
            "serial": serial,
            "cpu": cpu,
            "ram": ram,

            "product_no": model_info.get("product_no",""),
            "reg_model": model_info.get("reg_model",""),
            "reg_type": model_info.get("reg_type",""),
            "dpn": model_info.get("dpn",""),
            "input_power": model_info.get("input_power",""),
            "retail_model_guess": model_info.get("retail_model_guess",""),
            "notes": model_info.get("notes",""),

            "raw_json": json.dumps({
                "model_extracted": model_info,
                "serial_extracted": serial_info
            }, ensure_ascii=False)
        }

        # Log a provisional row into scans.csv
        flat, cols = ensure_columns(out)
        try:
            df = pd.DataFrame([flat], columns=cols)
            df.to_csv(
                DATA_CSV,
                mode=("a" if os.path.exists(DATA_CSV) else "w"),
                header=not os.path.exists(DATA_CSV),
                index=False
            )
        except Exception as e:
            print("ANALYZE_PAIR CSV WRITE ERROR:", repr(e), flush=True)

        return jsonify(out)

    except Exception as e:
        print("ANALYZE_PAIR ERROR:", repr(e))
        print(traceback.format_exc())
        return jsonify({"error": "analyze_pair_failed", "detail": str(e)}), 500

# --- Manual serial fallback (text only) ---

@app.route("/manual_serial_lookup", methods=["POST"])
def manual_serial_lookup():
    data = request.get_json(force=True, silent=True) or {}
    serial = (data.get("serial") or "").strip().upper()
    if not serial:
        return jsonify({"error":"missing_serial"}), 400

    looked = oem_lookup(serial) if ALLOW_LOOKUPS else {}

    return jsonify({
        "serial": serial,
        "manufacturer": looked.get("manufacturer",""),
        "model_no": looked.get("model_no",""),
        "cpu": looked.get("cpu",""),
        "ram": looked.get("ram",""),
        "raw_json": json.dumps({"oem_lookup": looked}, ensure_ascii=False)
    })

# --- Confirm/save + learn ---

@app.route("/confirm_row", methods=["POST"])
def confirm_row():
    """
    Confirms a row from the UI table.

    Behavior:
      - If scans.csv already has a row with the same 'time', we UPDATE that row.
      - Otherwise, we APPEND a new row.
      - Then we learn into models_db.json.
    """
    row = request.get_json(force=True, silent=True) or {}
    flat, cols = ensure_columns(row)
    row_time = flat.get("time", "")

    try:
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV, dtype=str, on_bad_lines="skip", engine="python")

            if "time" in df.columns and row_time:
                mask = df["time"] == row_time
                if mask.any():
                    for c in cols:
                        if c in df.columns:
                            df.loc[mask, c] = flat.get(c, "")
                        else:
                            df[c] = ""
                            df.loc[mask, c] = flat.get(c, "")
                else:
                    df = pd.concat([df, pd.DataFrame([flat], columns=cols)], ignore_index=True)
            else:
                df = pd.concat([df, pd.DataFrame([flat], columns=cols)], ignore_index=True)

            df.to_csv(DATA_CSV, index=False)
        else:
            pd.DataFrame([flat], columns=cols).to_csv(
                DATA_CSV,
                mode="w",
                header=True,
                index=False
            )
    except Exception as e:
        print("CONFIRM_ROW CSV WRITE ERROR:", repr(e), flush=True)

    model_no = (row.get("model_no") or row.get("retail_model_guess") or "").strip()
    manufacturer = (row.get("manufacturer") or "").strip()
    cpu = (row.get("cpu") or "").strip()
    ram = (row.get("ram") or "").strip()
    if model_no:
        learn_model_entry(model_no=model_no, manufacturer=manufacturer, cpu=cpu, ram=ram)

    return jsonify({"ok": True, "saved": flat.get("time","")})

# ===== View models_db.json in browser =====

@app.route("/view-models-db")
def view_models_db():
    if not os.path.exists(MODELS_DB_PATH):
        return (
            "<pre>"
            "models_db.json not found yet.\n\n"
            f"Expected path: {MODELS_DB_PATH}\n\n"
            "It will be created when you:\n"
            "  - Upload a model DB via /upload-db, or\n"
            "  - Confirm at least one row with a non-empty model_no.\n"
            "</pre>"
        )
    try:
        with open(MODELS_DB_PATH, "r", encoding="utf-8") as f:
            return "<pre>" + f.read() + "</pre>"
    except Exception as e:
        return f"<pre>Error reading models_db.json: {e}</pre>"

# ===== View scans.csv in browser (HTML table, last 200 rows) =====

@app.route("/view-scans")
def view_scans():
    if not os.path.exists(DATA_CSV):
        return (
            "<!doctype html><html><body style='font-family:system-ui;background:#0b0f17;color:#e9eef7;padding:20px'>"
            "<h2>scans.csv not found.</h2>"
            f"<p>Expected path: <code>{DATA_CSV}</code></p>"
            "<p>It is created when you confirm or analyze the first row.</p>"
            "</body></html>"
        )
    try:
        df = pd.read_csv(DATA_CSV, dtype=str, on_bad_lines="skip", engine="python")

        if "file_a" not in df.columns and "file" in df.columns:
            df["file_a"] = df["file"]

        df = df.tail(200)

        html_table = df.to_html(index=False, escape=False)
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>View Scans</title>"
            "<style>"
            "body{font-family:system-ui;background:#0b0f17;color:#e9eef7;padding:20px}"
            "h2{margin-bottom:12px}"
            "table{border-collapse:collapse;width:100%;background:#0f1522}"
            "th,td{border:1px solid #1b2740;padding:6px 8px;font-size:12px}"
            "th{background:#10192d}"
            "</style></head><body>"
            "<h2>Scans (last 200 rows)</h2>"
            + html_table +
            "</body></html>"
        )
    except Exception as e:
        return f"<pre>Error reading scans.csv: {e}</pre>", 500

# ===== NEW: JSON API for scans.csv (used by desktop to see phone scans) =====

@app.route("/api/scans", methods=["GET"])
def api_scans():
    """
    Returns the last 200 rows from scans.csv as JSON.

    Shape:
    {
      "rows": [
        {
          "time": "...",
          "order_no": "...",
          "file_a": "...",
          "file_b": "...",
          "manufacturer": "...",
          "model_no": "...",
          "serial": "...",
          "cpu": "...",
          "ram": "...",
          ...
        },
        ...
      ]
    }
    """
    if not os.path.exists(DATA_CSV):
        return jsonify({"rows": []})
    try:
        df = pd.read_csv(DATA_CSV, dtype=str, on_bad_lines="skip", engine="python")
        df = df.tail(200)
        df = df.fillna("")
        rows = df.to_dict(orient="records")
        return jsonify({"rows": rows})
    except Exception as e:
        print("API_SCANS ERROR:", repr(e))
        return jsonify({"error": "read_failed", "detail": str(e)}), 500

# ===== Upload DB page (unchanged) =====

INLINE_UPLOAD_DB_HTML = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upload Model Database (Fallback)</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:20px;background:#0b0f17;color:#e9eef7}
  .card{background:#0f1522;border:1px solid #1b2740;border-radius:12px;padding:14px;max-width:720px}
  button{padding:10px 14px;border-radius:8px;border:1px solid #3b5cb3;background:#0b1220;color:#e9eef7;cursor:pointer}
  button.primary{background:#2356ff;border-color:#2356ff;color:#fff}
  .hint{font-size:12px;color:#9fb0c7;margin-top:6px}
  a{color:#9ec5ff;text-decoration:none}
</style>
</head><body>
  <h2>Upload Model Database (Fallback)</h2>
  <div class="hint">You’re seeing this because <code>templates/upload_db.html</code> was not found.</div>
  <div class="card">
    <form id="upForm">
      <label>Choose .xlsx or .csv</label>
      <input type="file" id="file" name="file" accept=".xlsx,.csv" required>
      <div class="hint">Excel first sheet will be used. CSV must include headers.</div>
      <div style="margin-top:12px;display:flex;gap:10px">
        <button type="submit" class="primary">Upload & Build JSON</button>
        <a href="/">← Back to Intake</a>
      </div>
      <div id="status" class="hint" style="margin-top:10px"></div>
    </form>
  </div>
<script>
(function(){
  const form = document.getElementById('upForm');
  const status = document.getElementById('status');
  function setStatus(msg, ok){ status.textContent = msg; status.className = "hint " + (ok ? "ok" : "err"); }
  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const file = document.getElementById('file').files[0];
    if(!file){ setStatus('Choose a file first.', false); return; }
    setStatus('Uploading… building JSON…', true);
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch('/upload-db', { method:'POST', body: fd });
    const j = await r.json().catch(()=> ({}));
    if(!r.ok){ setStatus((j.detail || j.error || ('HTTP '+r.status)), false); return; }
    setStatus('OK — database updated. Saved to ' + (j.path || 'data/models_db.json') + ' (' + (j.count||0) + ' records).', true);
  });
})();
</script>
</body></html>"""

@app.route("/upload-db", methods=["GET"])
def upload_db_page():
    try:
        return render_template("upload_db.html")
    except TemplateNotFound:
        return INLINE_UPLOAD_DB_HTML

@app.route("/upload-db", methods=["POST"])
def upload_db_post():
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"no_file"}), 400
    name = (f.filename or "").lower()
    if not (name.endswith(".xlsx") or name.endswith(".csv")):
        return jsonify({"error":"unsupported_file_type"}), 400
    try:
        if name.endswith(".xlsx"):
            df = pd.read_excel(f, engine="openpyxl")
        else:
            df = pd.read_csv(f, dtype=str)

        df.columns = [c.strip().lower() for c in df.columns]
        alias = {"model":"model_no","model number":"model_no","modelname":"model_no","model_name":"model_no"}
        df.columns = [alias.get(c, c) for c in df.columns]

        if "model_no" not in df.columns:
            return jsonify({"error":"missing_column","detail":"model/model_no"}), 400
        if "manufacturer" not in df.columns:
            df["manufacturer"] = ""
        for c in ["cpu","ram","alt_names"]:
            if c not in df.columns:
                df[c] = ""

        latest_by_model = {}
        for _, row in df.iterrows():
            mdl = str(row.get("model_no") or "").strip()
            if not mdl:
                continue
            key = _norm(mdl)
            latest_by_model[key] = {
                "manufacturer": str(row.get("manufacturer") or "").strip(),
                "model_no": mdl,
                "cpu": str(row.get("cpu") or "").strip(),
                "ram": str(row.get("ram") or "").strip(),
                "alt_names": [a.strip() for a in str(row.get("alt_names") or "").split(",") if a.strip()]
            }

        out = list(latest_by_model.values())

        backup_path = ""
        if os.path.exists(MODELS_DB_PATH):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = MODELS_DB_PATH.replace(".json", f".{ts}.bak.json")
            shutil.copy2(MODELS_DB_PATH, backup_path)

        _save_models_db(out)
        return jsonify({"ok": True, "count": len(out), "path": MODELS_DB_PATH, "backup": backup_path or "(none)"})
    except Exception as e:
        print("UPLOAD-DB ERROR:", repr(e)); print(traceback.format_exc())
        return jsonify({"error":"build_failed","detail":str(e)}), 400

@app.route("/whereami")
def whereami():
    return jsonify({
        "app_root_path": app.root_path,
        "cwd": os.getcwd(),
        "template_folder": app.template_folder,
        "expected_template": os.path.join(app.template_folder or "templates", "upload_db.html"),
        "models_db_path": MODELS_DB_PATH
    })

if __name__ == "__main__":
    print(f"ALLOW_LOOKUPS={ALLOW_LOOKUPS}  LOOKUP_TIMEOUT={LOOKUP_TIMEOUT}s")
    print(f"ENABLE_MODEL_INFER={ENABLE_MODEL_INFER}")
    print(f"MODELS_DB_PATH={MODELS_DB_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=True)
