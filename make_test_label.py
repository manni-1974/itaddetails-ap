from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path

OUTDIR = Path(".")
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\Calibri.ttf",
    r"C:\Windows\Fonts\Tahoma.ttf",
]

def load_font(size=36):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def make_simple_white():
    """Black text on white label."""
    W, H = 1000, 500
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font_h = load_font(44)
    font_b = load_font(50)
    lines = [
        ("Manufacturer: ", "Dell"),
        ("Model: ", "Latitude 5310"),
        ("Serial No: ", "ABC1234XYZ"),
        ("Service Tag: ", "9B7G8F2"),
    ]
    y = 40
    for label, value in lines:
        draw.text((40, y), label, fill="black", font=font_h)
        draw.text((40 + draw.textlength(label, font=font_h), y),
                  value, fill="black", font=font_b)
        y += 70
    out = OUTDIR / "test_label.jpg"
    img.save(out, quality=95)
    print(f"✅ Created {out.resolve()}")

def make_simple_dark():
    """White text on black label."""
    W, H = 1000, 500
    img = Image.new("RGB", (W, H), "black")
    draw = ImageDraw.Draw(img)
    font_h = load_font(44)
    font_b = load_font(50)
    lines = [
        ("Manufacturer: ", "MSI"),
        ("Model No: ", "MS-17B1"),
        ("Serial: ", "SN7890QWE"),
    ]
    y = 40
    for label, value in lines:
        draw.text((40, y), label, fill="white", font=font_h)
        draw.text((40 + draw.textlength(label, font=font_h), y),
                  value, fill="white", font=font_b)
        y += 70
    out = OUTDIR / "test_label_dark.jpg"
    img.save(out, quality=95)
    print(f"✅ Created {out.resolve()}")

def make_dell_style_block():
    """Sticker-like layout with boxed border."""
    W, H = 1200, 700
    img = Image.new("RGB", (W, H), "#FAFAFA")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 20), (W-20, H-20)], outline="#222", width=4)
    title_font = load_font(56)
    label_font = load_font(40)
    value_font = load_font(46)
    small_font = load_font(32)
    title = "DELL MANUFACTURING LABEL"
    tw = draw.textlength(title, font=title_font)
    draw.text(((W - tw) / 2, 40), title, fill="#111", font=title_font)
    y = 140
    fields = [
        ("Manufacturer", "Dell"),
        ("Model", "Latitude 5310"),
        ("Service Tag", "9B7G8F2"),
        ("Express Service Code", "12345678901"),
        ("Product No.", "0N3-5310-CTO"),
        ("Serial Number", "ABC1234XYZ"),
    ]
    for k, v in fields:
        draw.text((60, y), f"{k}:", fill="#222", font=label_font)
        draw.text((400, y), v, fill="#000", font=value_font)
        y += 70
    draw.text((60, H-90), "Made in USA | 2025-11-03", fill="#333", font=small_font)
    out = OUTDIR / "dell_style_label.jpg"
    img.save(out, quality=95)
    print(f"✅ Created {out.resolve()}")

if __name__ == "__main__":
    make_simple_white()
    make_simple_dark()
    make_dell_style_block()
    print("All test labels created.")
