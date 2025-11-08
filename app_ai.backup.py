# app_ai.py — same UI/look; serial-first + OEM lookup + LLM spec guess (best-effort);
# + local JSON DB fallback + upload-db page (Excel/CSV -> data/models_db.json)

import os, re, io, json, time, base64, traceback
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pandas as pd

# ===== Optional OpenCV (unchanged) =====
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
MODEL     = os.getenv("VISION_MODEL", "gpt-4o-mini")  # same model for vision & tiny text task
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")

# Enable/disable serial → OEM lookup (best effort). Keep off if you don't want outbound hits.
ALLOW_LOOKUPS = os.getenv("ALLOW_LOOKUPS", "1") in ("1", "true", "True", "yes", "YES")
LOOKUP_TIMEOUT = float(os.getenv("LOOKUP_TIMEOUT", "8.0"))

DATA_CSV = "scans.csv"
UPLOADS  = "uploads"
os.makedirs(UPLOADS, exist_ok=True)

# >>> NEW: local DB file path
os.makedirs("data", exist_ok=True)
MODELS_DB = os.path.join("data", "models_db.json")

# Regulatory → retail hint table (unchanged)
REG_MODEL_TO_RETAIL = {"P131G": "Latitude 7410"}

# Known model spec hints (very small, safe defaults; used only if nothing else found)
SPEC_HINTS = {
    # Laptops
    "Latitude 7410": {
        "cpu": "Intel Core i5-10210U / i7-10610U (10th Gen, vPro optional)",
        "ram": "8 GB default, up to 32 GB DDR4-2666"
    },
    "Latitude 7410 2-in-1": {
        "cpu": "Intel Core i5-10210U / i7-10610U (10th Gen, vPro optional)",
        "ram": "8 GB default, up to 32 GB DDR4-2666"
    },
    # Desktops (ambiguous generations are labeled as 'varies by config')
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

# ---------- Image helpers (unchanged) ----------
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

# ---------- Regex (unchanged core + service tag + family) ----------
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

# ---------- Tiny helper: LLM text inference for CPU/RAM (best-effort) ----------
def ai_guess_specs(model_str: str) -> dict:
    model_str = (model_str or "").strip()
    if not API_KEY or not model_str:
        return {}

    # If we have a local hint, prefer it
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
        "If multiple CPU options exist, mention the generation/options concisely. "
        "Schema:\n"
        "{\n  \"cpu\": \"...\",\n  \"ram\": \"...\"\n}\n\n"
        f"Model: {model_str}\n"
        "Return ONLY the JSON."
    )

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=20)
        if r.status_code >= 400:
            try:
                print("LLM spec guess HTTP", r.status_code, "BODY:", r.text[:600])
            except Exception:
                pass
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
    except Exception as e:
        print("LLM spec guess error:", repr(e))
        return {}

# ---------- Vision call (unchanged) ----------
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
        "- Focus on the dark printed label (manufacturer, Model/Model No, S/N or Serial, P/N or Product No, DP/N, Regulatory Model/Type, input power). "
        "DO NOT invent values and DO NOT copy Service Tag.\n"
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
            if r.status_code >= 400:
                try:
                    print("OpenAI HTTP", r.status_code, "BODY:", r.text[:1200])
                except Exception:
                    pass
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

# ---------- OEM lookup helpers (best-effort; safe to fail) ----------
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
    return {}

def _lookup_hp_by_serial(sn: str):
    return {}

def oem_lookup(serial: str) -> dict:
    sn = (serial or "").strip().upper()
    if not sn:
        return {}
    if _is_dell_service_tag(sn):
        got = _lookup_dell_by_tag(sn)
        if got:
            if (not got.get("cpu") or not got.get("ram")) and got.get("model_no"):
                guessed = ai_guess_specs(got.get("model_no",""))
                if guessed.get("cpu"): got["cpu"] = guessed["cpu"]
                if guessed.get("ram"): got["ram"] = guessed["ram"]
            return got
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

# ---------- >>> NEW: Local DB helpers ----------
_models_cache = {"ts": 0, "data": []}

def _load_models_db():
    """Load models_db.json once every 10s (simple cache)."""
    try:
        if not os.path.exists(MODELS_DB):
            _models_cache["data"] = []
            _models_cache["ts"] = time.time()
            return []
        if time.time() - _models_cache["ts"] < 10 and _models_cache["data"]:
            return _models_cache["data"]
        with open(MODELS_DB, "r", encoding="utf-8") as f:
            jd = json.load(f)
        items = jd.get("models") if isinstance(jd, dict) else jd
        if not isinstance(items, list): items = []
        _models_cache["data"] = items
        _models_cache["ts"] = time.time()
        return items
    except Exception:
        return []

def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def fallback_from_db(manufacturer, model_no):
    """Return {"cpu","ram","model_no","manufacturer"} if we can match."""
    items = _load_models_db()
    man = _norm(manufacturer)
    mod = _norm(model_no)

    # 1) exact (manufacturer + model_no)
    for it in items:
        if _norm(it.get("manufacturer")) == man and _norm(it.get("model_no")) == mod:
            return {
                "manufacturer": it.get("manufacturer",""),
                "model_no": it.get("model_no",""),
                "cpu": it.get("cpu",""),
                "ram": it.get("ram",""),
            }

    # 2) same manufacturer, relaxed contains/startswith
    for it in items:
        if _norm(it.get("manufacturer")) != man:
            continue
        im = _norm(it.get("model_no"))
        if im == mod or (im and mod and (mod in im or im in mod)):
            return {
                "manufacturer": it.get("manufacturer",""),
                "model_no": it.get("model_no",""),
                "cpu": it.get("cpu",""),
                "ram": it.get("ram",""),
            }

    # 3) alt_names (semicolon/comma/pipe separated)
    for it in items:
        if _norm(it.get("manufacturer")) != man:
            continue
        alts = it.get("alt_names") or []
        if isinstance(alts, str):
            alts = re.split(r"[;,\|]", alts)
        for a in alts:
            if _norm(a) == mod:
                return {
                    "manufacturer": it.get("manufacturer",""),
                    "model_no": it.get("model_no",""),
                    "cpu": it.get("cpu",""),
                    "ram": it.get("ram",""),
                }
    return {}

# ---------- Fuse two pics (unchanged logic, with family fix) ----------
def fuse_pair(a: dict, b: dict) -> dict:
    out = {k: (a.get(k) or b.get(k) or "") for k in SCHEMA.keys()}
    combined_raw = " | ".join(filter(None, [a.get("raw_text",""), b.get("raw_text","")]))[:6000]

    # Service Tag preference when brand is Dell
    st = SERVICE_TAG_PAT.search(combined_raw)
    service_tag = st.group(1).upper() if st else ""

    if (a.get("manufacturer","") or b.get("manufacturer","")).lower().startswith("dell"):
        if service_tag:
            out["serial"] = service_tag
        elif out.get("serial") and not re.fullmatch(r"[A-Z0-9]{7}", out["serial"]):
            out["serial"] = ""

    # Family model refinement (e.g., "OptiPlex 7010")
    out["model_no"] = refine_model_with_family(out.get("model_no",""), combined_raw)

    # Retail guess via reg model
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
      - manual_serial (optional; when present we trust it first)
    Flow:
      1) Run vision on both images (unchanged).
      2) Fuse results (unchanged family fix).
      3) If manual_serial provided, override serial with it.
      4) If ALLOW_LOOKUPS and we have a serial, try OEM lookup for model/cpu/ram.
      5) If cpu/ram still blank but we have a marketing model, ask tiny LLM text call for typical CPU/RAM (best-effort).
      6) Try local DB fallback (if still blank).
      7) Return fused+enhanced JSON (saved on Confirm).
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

    # ---- Serial-first override ----
    if manual_serial:
        fused["serial"] = manual_serial

    # ---- OEM lookup (best-effort) ----
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

    # If CPU/RAM still blank but we have a model, ask LLM for concise typical specs (best-effort)
    if (not fused.get("cpu") or not fused.get("ram")) and fused.get("model_no"):
        guessed = ai_guess_specs(fused["model_no"])
        if not fused.get("cpu") and guessed.get("cpu"): fused["cpu"] = guessed["cpu"]
        if not fused.get("ram") and guessed.get("ram"): fused["ram"] = guessed["ram"]

    # Final family-name fix (keeps marketing names like “OptiPlex 7010”)
    fused["model_no"] = refine_model_with_family(fused.get("model_no",""), fused.get("raw_text",""))

    # >>> NEW: Local DB fallback (only if still blank)
    if (not fused.get("cpu") or not fused.get("ram")) and fused.get("model_no"):
        fb = fallback_from_db(fused.get("manufacturer",""), fused.get("model_no",""))
        if fb:
            if not fused.get("manufacturer") and fb.get("manufacturer"):
                fused["manufacturer"] = fb["manufacturer"]
            if not fused.get("model_no") and fb.get("model_no"):
                fused["model_no"] = fb["model_no"]
            if not fused.get("cpu") and fb.get("cpu"):
                fused["cpu"] = fb["cpu"]
            if not fused.get("ram") and fb.get("ram"):
                fused["ram"] = fb["ram"]

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

# ---------- >>> NEW: Uploader page (Excel/CSV -> models_db.json)
from werkzeug.utils import secure_filename

@app.route("/upload-db", methods=["GET"])
def upload_db_page():
    return render_template("upload_db.html")

@app.route("/upload-db", methods=["POST"])
def upload_db_post():
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"no file"}), 400

    name = secure_filename(f.filename or "")
    ext = (os.path.splitext(name)[1] or "").lower()
    if ext not in (".xlsx", ".csv"):
        return jsonify({"error":"use .xlsx or .csv"}), 400

    # Read to DataFrame
    try:
        if ext == ".xlsx":
            df = pd.read_excel(f, engine="openpyxl")
        else:
            df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": "read_failed", "detail": str(e)}), 400

    # Normalize column names
    cols = {c.strip().lower(): c for c in df.columns if isinstance(c, str)}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    c_mfr   = pick("manufacturer","mfr","brand","oem")
    c_model = pick("model_no","model","model number","name")
    c_cpu   = pick("cpu","processor")
    c_ram   = pick("ram","memory")
    c_alt   = pick("alt_names","alias","aliases","aka")

    if not c_mfr or not c_model:
        return jsonify({"error":"missing required columns",
                        "detail":"Need at least manufacturer + model/model_no headers"}), 400

    # Build rows
    out = []
    seen = set()
    for _, row in df.iterrows():
        mfr   = str(row.get(c_mfr, "")).strip()
        model = str(row.get(c_model, "")).strip()
        if not mfr or not model:
            continue
        key = (_norm(mfr), _norm(model))
        if key in seen:
            continue
        seen.add(key)

        cpu = str(row.get(c_cpu, "")).strip() if c_cpu else ""
        ram = str(row.get(c_ram, "")).strip() if c_ram else ""
        alt = row.get(c_alt, "") if c_alt else ""
        if isinstance(alt, float) and pd.isna(alt): alt = ""
        if isinstance(alt, str):
            alt_names = alt.strip()
        else:
            alt_names = ""

        out.append({
            "manufacturer": mfr,
            "model_no": model,
            "cpu": cpu,
            "ram": ram,
            "alt_names": alt_names
        })

    payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(out),
        "models": out
    }

    # Backup old file
    backup_path = ""
    try:
        if os.path.exists(MODELS_DB):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join("data", f"models_db.{ts}.json")
            os.replace(MODELS_DB, backup_path)
    except Exception:
        backup_path = "(failed to backup old file)"

    # Write new file
    try:
        with open(MODELS_DB, "w", encoding="utf-8") as fo:
            json.dump(payload, fo, ensure_ascii=False, indent=2)
    except Exception as e:
        return jsonify({"error":"write_failed","detail":str(e)}), 500

    # Reset cache
    _models_cache["data"] = out
    _models_cache["ts"] = time.time()

    return jsonify({"ok": True, "path": MODELS_DB, "count": len(out), "backup": backup_path or "(none)"}), 200

if __name__ == "__main__":
    print(f"ALLOW_LOOKUPS={ALLOW_LOOKUPS}  LOOKUP_TIMEOUT={LOOKUP_TIMEOUT}s")
    app.run(host="0.0.0.0", port=5000, debug=True)
