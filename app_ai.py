# + OEM lookup + vision model inference + DB + LLM spec guess
# Keeps all past features. Changes in this version:
#   - Upload DB dedupes by MODEL ONLY (manufacturer optional; latest row wins)
#   - Runtime db_lookup_specs() matches by MODEL ONLY
#   - /analyze_pair uses model-only DB lookup for final fill if blank

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

# Allow outbound OEM lookups + network timeouts
ALLOW_LOOKUPS = os.getenv("ALLOW_LOOKUPS", "1") in ("1","true","True","yes","YES")
LOOKUP_TIMEOUT = float(os.getenv("LOOKUP_TIMEOUT", "8.0"))

# Optional OEM configs (remain safe if blank)
LENOVO_API_BASE = os.getenv("LENOVO_API_BASE", "").rstrip("/")
LENOVO_API_KEY  = os.getenv("LENOVO_API_KEY", "")
HP_API_BASE     = os.getenv("HP_API_BASE", "").rstrip("/")
HP_API_KEY      = os.getenv("HP_API_KEY", "")

# Enable model inference from images when model not found
ENABLE_MODEL_INFER = os.getenv("ENABLE_MODEL_INFER", "1") in ("1","true","True","yes","YES")

DATA_CSV = "scans.csv"
UPLOADS  = "uploads"
DATA_DIR = "data"
MODELS_DB_PATH = os.getenv("MODELS_DB_PATH", os.path.join(DATA_DIR, "models_db.json"))
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Regulatory → retail hint table
REG_MODEL_TO_RETAIL = {"P131G": "Latitude 7410"}

# Known model spec hints (safe defaults)
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
        "cpu": "Varies by config (Ivy Bridge i5-3xxx in 2012 gen; newer '7010 Micro' uses 12th Gen)",
        "ram": "4–16 GB typical; check config"
    },
    "OptiPlex 7010 Micro": {
        "cpu": "Intel 12th Gen (e.g., i5-12500T) — varies by config",
        "ram": "8–32 GB DDR4 — varies by config"
    }
}

# ========== Flask ==========
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

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

def recent_log():
    if not os.path.exists(DATA_CSV):
        return "(no scans yet)"
    try:
        df = pd.read_csv(DATA_CSV, dtype=str, on_bad_lines="skip", engine="python")
        cols = [c for c in [
            "time","file_a","file_b","manufacturer","model_no","serial",
            "product_no","reg_model","reg_type","dpn","input_power","retail_model_guess"
        ] if c in df.columns]
        return df[cols].tail(10).to_string(index=False)
    except Exception as e:
        return f"(could not read CSV: {e})"

def ensure_columns(row):
    cols = [
        "time","file_a","file_b",
        "manufacturer","model_no","serial",
        "cpu","ram",
        "product_no","reg_model","reg_type","dpn","input_power",
        "retail_model_guess","notes","raw_json"
    ]
    out = {k:"" for k in cols}
    out.update(row)
    return out, cols

# ---------- Image helpers ----------
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
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0,0), 1.0)
    sharp = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

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

# ---------- Regex ----------
SERIAL_PAT    = re.compile(r"(?:S/N|SN|Serial(?:\s*No\.?)?|Serial\s*Number|CN-)\s*[:#]?\s*([A-Z0-9\-]{5,})", re.I)
MODELNO_PAT   = re.compile(r"(?:Model\s*No\.?|Model)\s*[:#]?\s*([A-Z0-9\-\._]+)", re.I)
DPN_PAT       = re.compile(r"\bDP/?N[:\s]*([A-Z0-9\-]+)\b", re.I)
PN_PAT        = re.compile(r"\b(P/?N|Product\s*No\.?)[:\s]*([A-Z0-9\-\._]+)\b", re.I)
REGMODEL_PAT  = re.compile(r"\bReg(?:ulatory)?\s*Model[:\s]*([A-Z0-9\-]+)\b", re.I)
REGTYPE_PAT   = re.compile(r"\bReg(?:ulatory)?\s*Type(?:\s*No\.?)?[:\s]*([A-Z0-9\-]+)\b", re.I)
INPUT_PAT     = re.compile(r"\b(1[29]\.?\d*V|20V)\s*[-–—]?\s*(\d{1,2}\.\d{1,2}A)\b", re.I)
SERVICE_TAG_PAT = re.compile(r"\b(?:ST|Service\s*Tag)\s*[:#]?\s*([A-Z0-9]{7})\b", re.I)
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
        if m: norm["model_no"] = m.group(1).strip()
    if not norm.get("serial"):
        m = SERIAL_PAT.search(raw)
        if m: norm["serial"] = m.group(1).strip().upper()
    if not norm.get("dpn"):
        m = DPN_PAT.search(raw)
        if m: norm["dpn"] = m.group(1).strip().upper()
    if not norm.get("product_no"):
        m = PN_PAT.search(raw)
        if m: norm["product_no"] = m.group(2).strip().upper()
    if not norm.get("reg_model"):
        m = REGMODEL_PAT.search(raw)
        if m: norm["reg_model"] = m.group(1).strip().upper()
    if not norm.get("reg_type"):
        m = REGTYPE_PAT.search(raw)
        if m: norm["reg_type"] = m.group(1).strip().upper()
    if not norm.get("input_power"):
        m = INPUT_PAT.search(raw)
        if m: norm["input_power"] = f"{m.group(1).upper()} {m.group(2)}"
    if not norm.get("retail_model_guess") and (norm.get("manufacturer","").lower() == "dell"):
        rm = (norm.get("reg_model") or "").upper()
        if rm in REG_MODEL_TO_RETAIL:
            norm["retail_model_guess"] = REG_MODEL_TO_RETAIL[rm]
    return norm

def refine_model_with_family(model_no: str, raw_text: str) -> str:
    fam = FAMILY_MODEL_PAT.search(raw_text or "")
    if not fam: return model_no
    brand = fam.group(1).strip()
    val   = fam.group(2).strip()
    val = re.sub(r"\b2\s*[- ]?\s*in\s*[- ]?1\b", "2-in-1", val, flags=re.I)
    cand = f"{brand} {val}"
    if (not model_no) or (len(model_no) <= 5 and " " not in model_no): return cand
    if brand.lower() in model_no.lower(): return model_no
    return cand

# ---------- LLM spec guess (text) ----------
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
    if OPENAI_PROJECT: headers["OpenAI-Project"] = OPENAI_PROJECT
    prompt = (
        "You are a hardware spec assistant. Given a laptop/desktop marketing model string, "
        "respond with a very short JSON object that guesses typical CPU family and RAM default for that model. "
        "If multiple CPU options exist, mention the generation/options concisely.\n"
        "{\n  \"cpu\": \"...\",\n  \"ram\": \"...\"\n}\n\n"
        f"Model: {model_str}\n"
        "Return ONLY the JSON."
    )
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=20)
        if r.status_code >= 400:
            return {}
        jd = r.json()
        txt = jd["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {}
        obj = json.loads(m.group(0))
        out = {}
        if isinstance(obj, dict):
            if obj.get("cpu"): out["cpu"] = str(obj["cpu"]).strip()
            if obj.get("ram"): out["ram"] = str(obj["ram"]).strip()
        return out
    except Exception:
        return {}

# ---------- Vision calls ----------
def call_openai_label_extract(img_path):
    if not API_KEY:
        raise RuntimeError("VISION_API_KEY is not set.")
    import requests
    raw = load_fix_exif(img_path)
    proc = preprocess_for_ocr(raw)
    inv  = inverted_contrast_variant(proc)
    data_proc = jpeg_data_url(proc, max_side=2400, quality=88)
    data_inv  = jpeg_data_url(inv,  max_side=2400, quality=88)
    data_raw  = jpeg_data_url(raw,  max_side=2400, quality=88)

    prompt = (
        "EXTRACT FIELDS FROM THE MANUFACTURING/REGULATORY LABEL ONLY (ignore any white Service Tag stickers):\n"
        "Return STRICT JSON matching this schema:\n"
        + json.dumps(SCHEMA, indent=2) + "\n\n"
        "Rules:\n"
        "- Focus on the dark printed label (manufacturer, Model/Model No, S/N or Serial, P/N or Product No, DP/N, Regulatory Model/Type, input power).\n"
        "- 'model_no' should be the explicit Model text on the label.\n"
        "- 'serial' is near S/N | SN | Serial.\n"
        "- Provide a short 'raw_text' dump for regex fallback.\n"
        "Return ONLY the JSON object."
    )

    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if OPENAI_PROJECT: headers["OpenAI-Project"] = OPENAI_PROJECT
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":data_proc}},
            {"type":"image_url","image_url":{"url":data_inv}},
            {"type":"image_url","image_url":{"url":data_raw}},
        ]}],
        "temperature": 0.0
    }

    import requests
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            jd = r.json()
            text = jd["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            if not m: raise ValueError(f"No JSON found in response: {text[:300]}")
            obj = json.loads(m.group(0))
            norm = {k: (obj.get(k) or "").strip() for k in SCHEMA.keys()}

            norm = parse_from_raw_text(norm)
            if norm["input_power"]:
                norm["input_power"] = re.sub(r"\s*[-–—]{1,2}\s*", " ", norm["input_power"]).strip()
            return norm
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (429,) or (code is not None and 500 <= code < 600):
                time.sleep(2 ** attempt); continue
            raise

def call_openai_model_infer(img_a_path, img_b_path):
    """
    If label OCR did not yield a usable model, ask vision to infer brand + retail model
    from both photos (logo, keyboard, casing cues, family lines).
    Returns dict like {"manufacturer": "...", "model_no": "..."} or {}.
    """
    if not (API_KEY and ENABLE_MODEL_INFER):
        return {}
    try:
        import requests
        imgA = load_fix_exif(img_a_path)
        imgB = load_fix_exif(img_b_path)
        a_url = jpeg_data_url(preprocess_for_ocr(imgA), max_side=2400, quality=88)
        b_url = jpeg_data_url(preprocess_for_ocr(imgB), max_side=2400, quality=88)
        prompt = (
            "From these two photos of a computer (label and/or device view), infer the most likely BRAND and retail MODEL name.\n"
            "If unsure between close variants, pick the single most likely marketing model (e.g., 'ThinkPad T480', 'HP EliteBook 840 G5', 'Latitude 7420').\n"
            "Return ONLY JSON with keys: manufacturer, model_no."
        )
        url = f"{API_BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        if OPENAI_PROJECT: headers["OpenAI-Project"] = OPENAI_PROJECT
        body = {
            "model": MODEL,
            "messages": [{"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":a_url}},
                {"type":"image_url","image_url":{"url":b_url}},
            ]}],
            "temperature": 0.1
        }
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code >= 400:
            return {}
        jd = r.json()
        txt = jd["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {}
        obj = json.loads(m.group(0))
        out = {}
        if isinstance(obj, dict):
            man = (obj.get("manufacturer") or "").strip()
            mod = (obj.get("model_no") or "").strip()
            if man or mod:
                out["manufacturer"] = man
                out["model_no"] = mod
        return out
    except Exception:
        return {}

# ---------- OEM lookup helpers ----------
def _is_dell_service_tag(tag: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{7}", (tag or "").strip().upper()))

def _lookup_dell_by_tag(tag: str):
    """ Public Dell support endpoints (no key). May rate-limit. """
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
            r = requests.get(url, timeout=LOOKUP_TIMEOUT, headers={"Accept": "application/json"})
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, dict):
                summaries = data.get("assetSummaries") or data.get("AssetSummaries")
                if isinstance(summaries, list) and summaries:
                    s0 = summaries[0]
                    model = s0.get("productLineDescription") or s0.get("Model") or s0.get("productId") or ""
                    if model:
                        out["manufacturer"] = "DELL"
                        out["model_no"] = str(model).strip()
                        cpu = s0.get("cpu") or s0.get("cpuDescription") or ""
                        ram = s0.get("memory") or s0.get("ram") or ""
                        if cpu: out["cpu"] = str(cpu).strip()
                        if ram: out["ram"] = str(ram).strip()
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

def _lookup_lenovo_by_serial(sn: str):
    """
    Placeholder: Lenovo often requires PSREF or warranty endpoints that need keys or extra params.
    If you have an internal gateway, set LENOVO_API_BASE/LENOVO_API_KEY and implement below.
    Safe no-op otherwise.
    """
    out = {}
    sn = (sn or "").strip().upper()
    if not (ALLOW_LOOKUPS and LENOVO_API_BASE and LENOVO_API_KEY and sn):
        return out
    try:
        import requests
        r = requests.get(
            f"{LENOVO_API_BASE}/lookup",
            params={"serial": sn},
            headers={"X-API-Key": LENOVO_API_KEY, "Accept":"application/json"},
            timeout=LOOKUP_TIMEOUT
        )
        if r.status_code != 200:
            return out
        data = r.json() if r.content else {}
        if isinstance(data, dict):
            for k in ("manufacturer","model_no","cpu","ram"):
                if data.get(k): out[k] = str(data[k]).strip()
        return out
    except Exception:
        return {}

def _lookup_hp_by_serial(sn: str):
    """
    Placeholder: HP warranty/spec APIs typically require auth.
    If you have a gateway, set HP_API_BASE/HP_API_KEY and implement below.
    """
    out = {}
    sn = (sn or "").strip().upper()
    if not (ALLOW_LOOKUPS and HP_API_BASE and HP_API_KEY and sn):
        return out
    try:
        import requests
        r = requests.get(
            f"{HP_API_BASE}/lookup",
            params={"serial": sn},
            headers={"X-API-Key": HP_API_KEY, "Accept":"application/json"},
            timeout=LOOKUP_TIMEOUT
        )
        if r.status_code != 200:
            return out
        data = r.json() if r.content else {}
        if isinstance(data, dict):
            for k in ("manufacturer","model_no","cpu","ram"):
                if data.get(k): out[k] = str(data[k]).strip()
        return out
    except Exception:
        return {}

def oem_lookup(serial: str) -> dict:
    """ Dispatch to vendor-specific lookups. Best-effort & safe to fail. """
    sn = (serial or "").strip().upper()
    if not sn:
        return {}
    # Dell first (exact 7-char)
    if _is_dell_service_tag(sn):
        got = _lookup_dell_by_tag(sn)
        if got:
            if (not got.get("cpu") or not got.get("ram")) and got.get("model_no"):
                guessed = ai_guess_specs(got.get("model_no",""))
                if guessed.get("cpu"): got["cpu"] = guessed["cpu"]
                if guessed.get("ram"): got["ram"] = guessed["ram"]
            return got
    # Other vendors
    for f in (_lookup_lenovo_by_serial, _lookup_hp_by_serial):
        try:
            got = f(sn)
            if got:
                if (not got.get("cpu") or not got.get("ram")) and got.get("model_no"):
                    guessed = ai_guess_specs(got.get("model_no",""))
                    if guessed.get("cpu"): got["cpu"] = guessed["cpu"]
                    if guessed.get("ram"): got["ram"] = guessed["ram"]
                return got
        except Exception:
            pass
    return {}

# ---------- Local DB ----------
def _load_models_db():
    try:
        if os.path.exists(MODELS_DB_PATH):
            with open(MODELS_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print("models_db load error:", repr(e))
    return []

def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def db_lookup_specs(model_name: str) -> dict:
    """
    Model-only lookup (manufacturer ignored).
    Returns {"cpu": "...", "ram": "..."} if found, else {}.
    Matches either model_no or any alt_names entry (case/spacing-insensitive).
    """
    model_name_n = _norm(model_name)
    if not model_name_n:
        return {}

    records = _load_models_db()
    for rec in records:
        mdl = _norm(rec.get("model_no", "") or rec.get("model", ""))
        alts = rec.get("alt_names") or []
        if not isinstance(alts, list):
            alts = []

        if (mdl and mdl == model_name_n) or any(_norm(a) == model_name_n for a in alts):
            out = {}
            cpu = (rec.get("cpu") or "").strip()
            ram = (rec.get("ram") or "").strip()
            if cpu: out["cpu"] = cpu
            if ram: out["ram"] = ram
            return out
    return {}

# ---------- Fuse two pics ----------
def fuse_pair(a: dict, b: dict) -> dict:
    out = {k: (a.get(k) or b.get(k) or "") for k in SCHEMA.keys()}
    combined_raw = " | ".join(filter(None, [a.get("raw_text",""), b.get("raw_text","")]))[:6000]

    st = SERVICE_TAG_PAT.search(combined_raw)
    service_tag = st.group(1).upper() if st else ""

    # Only auto-enforce 7-char tag for Dell; otherwise allow arbitrary serials
    if (a.get("manufacturer","") or b.get("manufacturer","")).lower().startswith("dell"):
        if service_tag:
            out["serial"] = service_tag
        elif out.get("serial") and not re.fullmatch(r"[A-Z0-9]{7}", out["serial"]):
            out["serial"] = ""

    # Family model refinement
    out["model_no"] = refine_model_with_family(out.get("model_no",""), combined_raw)

    # Retail guess via reg model (Dell)
    if not out.get("retail_model_guess") and (out.get("manufacturer","").lower() == "dell"):
        rm = (out.get("reg_model") or "").upper()
        if rm in REG_MODEL_TO_RETAIL:
            out["retail_model_guess"] = REG_MODEL_TO_RETAIL[rm]

    out["raw_text"] = combined_raw
    return out

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html", log=recent_log())

@app.route("/selftest")
def selftest():
    import requests
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if OPENAI_PROJECT: headers["OpenAI-Project"] = OPENAI_PROJECT
    body = {"model": MODEL, "messages": [{"role":"user","content":"Reply 'ok'."}], "temperature": 0.0}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code >= 400:
            return jsonify({"ok": False, "status": r.status_code, "body": r.text}), 500
        txt = r.json()["choices"][0]["message"]["content"]
        return jsonify({"ok": True, "reply": txt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_single():
    f = request.files.get("file")
    if not f: return "No file uploaded", 400
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{f.filename}"
    path = os.path.join(UPLOADS, filename)
    f.save(path)
    try:
        img = load_fix_exif(path); img.save(path, "JPEG", quality=90, optimize=True)
    except Exception: pass
    try:
        extracted = call_openai_label_extract(path)
    except Exception as e:
        print("UPLOAD ERROR:", repr(e)); print(traceback.format_exc())
        return jsonify({"error":"vision_llm_failed", "detail": str(e)}), 400
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "file_a": filename, "file_b": "",
        "manufacturer": extracted.get("manufacturer",""),
        "model_no": extracted.get("model_no",""),
        "serial": extracted.get("serial",""),
        "cpu": "", "ram": "",
        "product_no": extracted.get("product_no",""),
        "reg_model": extracted.get("reg_model",""),
        "reg_type": extracted.get("reg_type",""),
        "dpn": extracted.get("dpn",""),
        "input_power": extracted.get("input_power",""),
        "retail_model_guess": extracted.get("retail_model_guess",""),
        "notes": extracted.get("notes",""),
        "raw_json": json.dumps({"extracted": extracted}, ensure_ascii=False)
    }
    flat, cols = ensure_columns(row)
    pd.DataFrame([flat], columns=cols).to_csv(
        DATA_CSV,
        mode=("a" if os.path.exists(DATA_CSV) else "w"),
        header=not os.path.exists(DATA_CSV),
        index=False
    )
    return jsonify(row)

@app.route("/analyze_pair", methods=["POST"])
def analyze_pair():
    """
    Accepts:
      - fileA, fileB  (images)
      - manual_serial (optional; any length; Dell OEM only if 7-char)
    Flow:
      1) Label vision on both images.
      2) Fuse results.
      3) If manual_serial provided, override serial with it (any length).
      4) If ALLOW_LOOKUPS and serial, try OEM lookup (Dell/Lenovo/HP). (Lenovo/HP no-op unless configured)
      5) If cpu/ram blank but we have a model ⇒ DB fallback (model-only).
      6) If model still blank ⇒ vision-based model inference from the two images.
      7) If still cpu/ram blank but model known ⇒ tiny LLM text guess.
      8) Return fused+enhanced JSON.
    """
    fA = request.files.get("fileA")
    fB = request.files.get("fileB")
    manual_serial = (request.form.get("manual_serial") or "").strip().upper()

    if not fA or not fB:
        return jsonify({"error":"need both photos"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nameA = f"{ts}_A_{fA.filename or 'a.jpg'}"
    nameB = f"{ts}_B_{fB.filename or 'b.jpg'}"
    pA = os.path.join(UPLOADS, nameA); fA.save(pA)
    pB = os.path.join(UPLOADS, nameB); fB.save(pB)
    try:
        img = load_fix_exif(pA); img.save(pA, "JPEG", quality=90, optimize=True)
    except Exception: pass
    try:
        img = load_fix_exif(pB); img.save(pB, "JPEG", quality=90, optimize=True)
    except Exception: pass

    try:
        a = call_openai_label_extract(pA)
        b = call_openai_label_extract(pB)
        fused = fuse_pair(a, b)
    except Exception as e:
        print("ANALYZE ERROR:", repr(e)); print(traceback.format_exc())
        return jsonify({"error":"vision_llm_failed","detail":str(e)}), 400

    # Serial override (any length)
    if manual_serial:
        fused["serial"] = manual_serial

    # OEM lookup (best-effort)
    looked = {}
    if ALLOW_LOOKUPS and fused.get("serial"):
        try:
            looked = oem_lookup(fused["serial"])
        except Exception as e:
            print("OEM LOOKUP ERROR:", repr(e))

    # Merge OEM info (prefer OEM where present)
    if looked:
        if looked.get("manufacturer"): fused["manufacturer"] = looked["manufacturer"]
        if looked.get("model_no"):     fused["model_no"]     = looked["model_no"]
        if looked.get("cpu"):          fused["cpu"]          = looked["cpu"]
        if looked.get("ram"):          fused["ram"]          = looked["ram"]

    # If model still blank, try vision model inference on pair of photos
    if ENABLE_MODEL_INFER and not (fused.get("model_no") or fused.get("retail_model_guess")):
        inferred = call_openai_model_infer(pA, pB)
        if inferred:
            if inferred.get("manufacturer") and not fused.get("manufacturer"):
                fused["manufacturer"] = inferred["manufacturer"]
            if inferred.get("model_no") and not fused.get("model_no"):
                fused["model_no"] = inferred["model_no"]

    # If CPU/RAM blank but we have a model, try local DB (model-only)
    if (not fused.get("cpu") or not fused.get("ram")):
        model_for_db = fused.get("model_no") or fused.get("retail_model_guess")
        if model_for_db:
            dbhit = db_lookup_specs(model_for_db)
            if not fused.get("cpu") and dbhit.get("cpu"): fused["cpu"] = dbhit["cpu"]
            if not fused.get("ram") and dbhit.get("ram"): fused["ram"] = dbhit["ram"]

    # If still blank but we have a model, tiny LLM text guess
    if (not fused.get("cpu") or not fused.get("ram")) and fused.get("model_no"):
        guessed = ai_guess_specs(fused["model_no"])
        if not fused.get("cpu") and guessed.get("cpu"): fused["cpu"] = guessed["cpu"]
        if not fused.get("ram") and guessed.get("ram"): fused["ram"] = guessed["ram"]

    # Final family-name fix
    fused["model_no"] = refine_model_with_family(fused.get("model_no",""), fused.get("raw_text",""))

    out = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "file_a": nameA, "file_b": nameB,
        "manufacturer": fused.get("manufacturer",""),
        "model_no": fused.get("model_no",""),
        "serial": fused.get("serial",""),
        "cpu": fused.get("cpu",""),
        "ram": fused.get("ram",""),
        "product_no": fused.get("product_no",""),
        "reg_model": fused.get("reg_model",""),
        "reg_type": fused.get("reg_type",""),
        "dpn": fused.get("dpn",""),
        "input_power": fused.get("input_power",""),
        "retail_model_guess": fused.get("retail_model_guess",""),
        "notes": fused.get("notes",""),
        "raw_json": json.dumps({"a":a,"b":b,"fused":fused,"oem":looked}, ensure_ascii=False)
    }
    return jsonify(out)

@app.route("/confirm_row", methods=["POST"])
def confirm_row():
    row = request.get_json(force=True, silent=True) or {}
    flat, cols = ensure_columns(row)
    pd.DataFrame([flat], columns=cols).to_csv(
        DATA_CSV,
        mode=("a" if os.path.exists(DATA_CSV) else "w"),
        header=not os.path.exists(DATA_CSV),
        index=False
    )
    return jsonify({"ok": True, "saved": flat.get("time","")})

# ===== Upload DB page (GET) — inline fallback =====
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
    setStatus('OK — database updated (deduped by model only). Saved to ' + (j.path || 'data/models_db.json') + ' (' + (j.count||0) + ' records).', true);
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

# ===== Upload DB builder (MODEL-ONLY DEDUPE) =====
@app.route("/upload-db", methods=["POST"])
def upload_db_post():
    """
    Excel/CSV → data/models_db.json
    Dedup policy: **by model only** (manufacturer ignored). Latest row wins.
    Required column: model or model_no
    Optional: manufacturer, cpu, ram, alt_names (comma-separated)
    """
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no_file"}), 400

    name = (f.filename or "").lower()
    if not (name.endswith(".xlsx") or name.endswith(".csv")):
        return jsonify({"error": "unsupported_file_type"}), 400

    try:
        if name.endswith(".xlsx"):
            df = pd.read_excel(f, engine="openpyxl")
        else:
            df = pd.read_csv(f, dtype=str)

        # normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        alias = {
            "model":"model_no",
            "model number":"model_no",
            "modelname":"model_no",
            "model_name":"model_no"
        }
        df.columns = [alias.get(c, c) for c in df.columns]

        if "model_no" not in df.columns:
            return jsonify({"error":"missing_column", "detail":"model/model_no"}), 400
        if "manufacturer" not in df.columns: df["manufacturer"] = ""
        if "cpu" not in df.columns: df["cpu"] = ""
        if "ram" not in df.columns: df["ram"] = ""
        if "alt_names" not in df.columns: df["alt_names"] = ""

        # Deduplicate by model only (latest row wins)
        by_model = {}
        order = []
        for _, row in df.iterrows():
            mdl = str(row.get("model_no") or "").strip()
            if not mdl:
                continue
            key = _norm(mdl)

            alts_raw = row.get("alt_names")
            if isinstance(alts_raw, str) and alts_raw.strip():
                alt_list = [a.strip() for a in alts_raw.split(",") if a.strip()]
            elif isinstance(alts_raw, list):
                alt_list = [str(a).strip() for a in alts_raw if str(a).strip()]
            else:
                alt_list = []

            rec = {
                "manufacturer": str(row.get("manufacturer") or "").strip(),
                "model_no": mdl,
                "cpu": str(row.get("cpu") or "").strip(),
                "ram": str(row.get("ram") or "").strip(),
                "alt_names": alt_list
            }
            by_model[key] = rec
            if key in order:
                order.remove(key)
            order.append(key)

        out = [by_model[k] for k in order]

        # Backup and write JSON
        backup_path = ""
        if os.path.exists(MODELS_DB_PATH):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = MODELS_DB_PATH.replace(".json", f".{ts}.bak.json")
            shutil.copy2(MODELS_DB_PATH, backup_path)

        with open(MODELS_DB_PATH, "w", encoding="utf-8") as f2:
            json.dump(out, f2, ensure_ascii=False, indent=2)

        return jsonify({"ok": True, "count": len(out), "path": MODELS_DB_PATH, "backup": backup_path or "(none)"})
    except Exception as e:
        print("UPLOAD-DB ERROR:", repr(e))
        print(traceback.format_exc())
        return jsonify({"error":"build_failed", "detail": str(e)}), 400

@app.route("/whereami")
def whereami():
    return jsonify({
        "app_root_path": app.root_path,
        "template_folder": app.template_folder,
        "expected_template": os.path.join(app.template_folder or "templates", "upload_db.html"),
        "cwd": os.getcwd(),
        "models_db_path": MODELS_DB_PATH
    })

if __name__ == "__main__":
    print(f"ALLOW_LOOKUPS={ALLOW_LOOKUPS}  LOOKUP_TIMEOUT={LOOKUP_TIMEOUT}s")
    print(f"ENABLE_MODEL_INFER={ENABLE_MODEL_INFER}")
    print(f"MODELS_DB_PATH={MODELS_DB_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=True)
