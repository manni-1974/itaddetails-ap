import os, re
from datetime import datetime
from itertools import product
from flask import Flask, request, render_template_string, jsonify, send_from_directory
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
import pandas as pd

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
DATA_CSV = "scans.csv"
UPLOAD_FOLDER = "uploads"
DEBUG_TXT = "last_ocr.txt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Scanner</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background:#f7f7f7; }
h2 { margin: 0 0 12px; }
input[type=file], input[type=submit] { display:block; width:100%; margin-top:10px; padding:12px; font-size:18px; border-radius:8px; border:1px solid #ccc; }
input[type=submit]{ background:#0078e7; color:#fff; border:none; font-weight:600; }
pre { white-space: pre-wrap; background:#fff; padding:12px; border-radius:8px; }
small {color:#666;}
</style>
</head>
<body>
  <h2>Upload a Laptop Label Photo</h2>
  <form method="post" enctype="multipart/form-data" action="/upload">
    <input type="file" name="file" accept="image/*" capture="environment">
    <input type="submit" value="Upload & Scan">
  </form>
  <p><small>If camera/upload doesn’t appear on mobile, try Chrome and plain HTTP.</small></p>
  <hr>
  <h3>Recent Scans (last 10)</h3>
  <pre>{{log}}</pre>
  <p><a href="/debug">View last OCR attempts (debug)</a> • <a href="/recent">Recent (JSON)</a></p>
</body>
</html>
"""

# ---------------- OCR helpers ----------------

def clean_text(s: str) -> str:
    # Normalize common OCR mix-ups and whitespace
    s = s.replace("|", "I")
    return "\n".join(" ".join(line.split()) for line in s.splitlines() if line.strip())

def preprocess_variants(img: Image.Image):
    """Generate multiple image variants for better OCR robustness."""
    variants = []
    # raw RGB
    variants.append(("raw", img.copy()))
    # grayscale -> invert -> contrast -> upscale -> median -> thresholds
    g = ImageOps.grayscale(img)
    inv = ImageOps.invert(g)
    inv_con = ImageEnhance.Contrast(inv).enhance(2.0)
    w, h = inv_con.size
    inv_con = inv_con.resize((int(w*2), int(h*2)), Image.LANCZOS)
    inv_con = inv_con.filter(ImageFilter.MedianFilter(size=3))
    variants.append(("inverted_contrast_upscale", inv_con))
    for t in (140, 170, 200):
        bw = inv_con.point(lambda p, T=t: 255 if p > T else 0)
        variants.append((f"bw_{t}", bw))
    return variants

def parse_fields_from_text(raw_text: str):
    """
    Parse fields using line-aware patterns.
    Returns: manu, model, serial, service_tag, express_code, product_no
    """
    text = clean_text(raw_text)
    manu = model = serial = service_tag = express = product_no = ""

    brands = ['Lenovo','Dell','HP','Hewlett-Packard','Microsoft','Asus','Acer','Samsung','MSI','Toshiba']
    for b in brands:
        if re.search(rf"\b{re.escape(b)}\b", text, re.I):
            manu = b
            break

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def find_value(label_regex):
        rx = re.compile(label_regex, re.I)
        for ln in lines:
            m = rx.search(ln)
            if m:
                if m.lastindex and m.lastindex >= 2 and m.group(2):
                    return m.group(2).strip()
                if m.lastindex and m.group(1):
                    return m.group(1).strip()
        return ""

    # Model / Model No / Product Name
    if not model:
        v = find_value(r"(?:^|\b)(Model(?:\s*No\.?)?|Product\s*Name)\s*[:\-]?\s*(.+)$")
        if v:
            # avoid over-capture into next label
            v = re.sub(r"\b(Service\s*Tag|Express\s*Service\s*Code|Serial|S/?N)\b.*$", "", v, flags=re.I).strip()
            model = v

    # Dell family name
    if not model:
        m = re.search(r"\b(Latitude\s+[A-Za-z0-9\-]+|XPS\s+[A-Za-z0-9\-]+|Precision\s+[A-Za-z0-9\-]+)\b", text, re.I)
        if m: model = m.group(1).strip()

    # Lenovo family name
    if not model:
        m = re.search(r"\b(ThinkPad\s+[A-Z0-9\-]+|IdeaPad\s+[A-Z0-9\-]+|ThinkBook\s+[A-Z0-9\-]+)\b", text, re.I)
        if m: model = m.group(1).strip()

    # MSI model codes (e.g., MS-17B1)
    if not model:
        m = re.search(r"\b(MS[\-]?\d{2,4}[A-Za-z0-9]*)\b", text, re.I)
        if m: model = m.group(1).strip()

    # Serial Number / S/N (generic)
    if not serial:
        serial = find_value(r"(?:^|\b)(Serial(?:\s*Number)?|S[/\-]?\s?N)\s*[:\-]?\s*([A-Za-z0-9\-]+)")

    # Dell Service Tag
    if not service_tag:
        service_tag = find_value(r"(?:^|\b)(Service\s*Tag)\s*[:\-]?\s*([A-Za-z0-9]{5,12})")

    # Express Service Code — tolerant to OCR typos like "Cot", "C0de", '#'
    if not express:
        # Prefer line-wise scan: any line mentioning "Express"
        for ln in lines:
            if re.search(r'Express', ln, re.I):
                m = re.search(r'([0-9]{6,20})', ln)  # first long digit run
                if m:
                    express = m.group(1)
                    break
    if not express:
        # Fallback: digits near "Express" in whole text
        m = re.search(r'Express[\sA-Za-z#:/\-]{0,30}([0-9]{6,20})', text, re.I)
        if m:
            express = m.group(1)

    # Product No / ProdID
    if not product_no:
        product_no = find_value(r"(?:^|\b)(Product\s*No\.?|ProdID)\s*[:\-]?\s*([A-Za-z0-9\-\#]+)")

    if not manu and "MSI" in text.upper():
        manu = "MSI"

    # If no explicit serial but a Service Tag exists, use as fallback
    if not serial and service_tag:
        serial = service_tag

    def tidy(s): return s.strip(":- ").replace("Model ", "") if s else s
    model = tidy(model)
    serial = tidy(serial)
    service_tag = tidy(service_tag)
    express = tidy(express)
    product_no = tidy(product_no)

    return manu, model, serial, service_tag, express, product_no

def score_fields(*vals):
    return sum(1 for v in vals if v)

def try_zoom_for_sn(img: Image.Image):
    """
    Scan small crops for S/N-like patterns (e.g., S/N, SN:, Serial).
    Returns serial if found, else "".
    """
    W, H = img.size
    # Try 3×3-ish bottom-heavy grid of crops (serials often sit lower on labels)
    grid_x = [0.0, 0.33, 0.66]
    grid_y = [0.33, 0.66, 0.75]
    for gx, gy in product(grid_x, grid_y):
        x0 = int(gx * W); y0 = int(gy * H)
        x1 = min(W, x0 + int(0.45 * W))
        y1 = min(H, y0 + int(0.28 * H))
        crop = img.crop((x0, y0, x1, y1))
        for var_label, im in preprocess_variants(crop):
            for cfg in ["--oem 3 --psm 6", "--oem 3 --psm 7", "--oem 3 --psm 11"]:
                text = pytesseract.image_to_string(im, lang="eng", config=cfg)
                t = clean_text(text)
                m = re.search(r"(Serial(?:\s*Number)?|S[/\-]?\s?N)\s*[:\-]?\s*([A-Za-z0-9\-]{6,})", t, re.I)
                if m:
                    return m.group(2).strip(":- ")
    return ""

def run_ocr(img: Image.Image):
    variants = preprocess_variants(img)
    configs = ["--oem 3 --psm 6", "--oem 3 --psm 7", "--oem 3 --psm 11", "--oem 1 --psm 6"]

    best = (0, "", "", "", "", "", "", "", "")  # score, manu, model, serial, st, esc, pno, text, var|cfg
    logs = []

    for vlabel, im in variants:
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(im, lang="eng", config=cfg)
            except Exception as e:
                logs.append(f"[{vlabel} | {cfg}] ERROR: {e}")
                continue

            text_c = clean_text(text)
            manu, model, serial, st, esc, pno = parse_fields_from_text(text_c)
            s = score_fields(manu, model, serial, st)
            logs.append(f"[{vlabel} | {cfg}] score={s} manu='{manu}' model='{model}' serial='{serial}' st='{st}' esc='{esc}' pno='{pno}' TEXT='{text_c[:200]}...'")
            if s > best[0]:
                best = (s, manu, model, serial, st, esc, pno, text_c, f"{vlabel}|{cfg}")

    # Tiny-text serial fallback (zoom crops)
    # Only try if we still don't have a serial
    if not best[3]:
        serial_zoom = try_zoom_for_sn(img)
        if serial_zoom:
            best = (best[0], best[1], best[2], serial_zoom, best[4], best[5], best[6], best[7], best[8])

    return best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], "\n".join(logs)

# ---------------- Routes ----------------

@app.route("/")
def home():
    log = ""
    if os.path.exists(DATA_CSV):
        try:
            df = pd.read_csv(DATA_CSV, engine="python", on_bad_lines="skip", dtype=str)
            cols = [c for c in ["time","file","manufacturer","model","serial","service_tag","express_service_code","product_no","ocr_variant","ocr_config"] if c in df.columns]
            if cols:
                log = df[cols].tail(10).to_string(index=False)
            else:
                log = df.tail(10).to_string(index=False)
        except Exception as e:
            log = f"(Could not read {DATA_CSV}: {e})"
    return render_template_string(HTML, log=log)

@app.route("/recent")
def recent():
    if not os.path.exists(DATA_CSV):
        return jsonify([])
    df = pd.read_csv(DATA_CSV, engine="python", on_bad_lines="skip", dtype=str)
    return jsonify(df.tail(10).to_dict(orient="records"))

@app.route("/debug")
def debug_last():
    if os.path.exists(DEBUG_TXT):
        return send_from_directory(".", DEBUG_TXT, as_attachment=False)
    return "No debug file yet. Upload a label first.", 200

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return "No file uploaded", 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{f.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    img = Image.open(path).convert("RGB")

    manu, model, serial, st, esc, pno, text_used, varcfg, attempts_log = run_ocr(img)

    # Write debug attempts
    with open(DEBUG_TXT, "w", encoding="utf-8") as dbg:
        dbg.write(f"IMAGE: {filename}\nBEST: {varcfg}\nTEXT:\n{text_used}\n\nATTEMPTS:\n{attempts_log}\n")

    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "file": filename,
        "manufacturer": manu,
        "model": model,
        "serial": serial,
        "service_tag": st,
        "express_service_code": esc,
        "product_no": pno,
        "raw_text": text_used,
        "ocr_variant": varcfg.split("|",1)[0] if "|" in varcfg else varcfg,
        "ocr_config": varcfg.split("|",1)[1] if "|" in varcfg else ""
    }

    df_row = pd.DataFrame([row])
    if os.path.exists(DATA_CSV):
        df_row.to_csv(DATA_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(DATA_CSV, index=False)

    return jsonify(row)

if __name__ == "__main__":
    # host=0.0.0.0 so your phone can connect over Wi-Fi
    app.run(host="0.0.0.0", port=5000, debug=True)
