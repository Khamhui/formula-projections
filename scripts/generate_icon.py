"""Generate app icon for F1 Predictions.app.

Usage:
    python scripts/generate_icon.py
"""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
OUTPUT_DIR = Path(__file__).parent.parent / "F1 Predictions.app" / "Contents" / "Resources"


def generate_icon():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 40
    radius = 200

    # Dark background
    draw.rounded_rectangle(
        [margin, margin, SIZE - margin, SIZE - margin],
        radius=radius,
        fill=(12, 12, 18, 255),
    )

    # Red racing stripe (left edge accent)
    draw.rectangle([margin, 300, margin + 12, 700], fill=(220, 40, 40, 255))

    # Checkered flag (bottom-right)
    cs = 28
    ox, oy = 660, 760
    for row in range(4):
        for col in range(8):
            x, y = ox + col * cs, oy + row * cs
            if x + cs > SIZE - margin or y + cs > SIZE - margin:
                continue
            color = (255, 255, 255, 200) if (row + col) % 2 == 0 else (50, 50, 60, 200)
            draw.rectangle([x, y, x + cs, y + cs], fill=color)

    # Load font
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 260)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
    except (OSError, IOError):
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # "F1" — red F, white 1
    draw.text((200, 260), "F", fill=(220, 40, 40, 255), font=font_big)
    draw.text((430, 260), "1", fill=(255, 255, 255, 255), font=font_big)

    # "PREDICTIONS" subtitle
    draw.text((200, 570), "PREDICTIONS", fill=(100, 100, 120, 255), font=font_small)

    # Save PNG
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "AppIcon.png"
    img.save(png_path, "PNG")
    print(f"PNG: {png_path}")

    # Convert to .icns
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            iconset = Path(tmpdir) / "AppIcon.iconset"
            iconset.mkdir()
            for size in [16, 32, 64, 128, 256, 512, 1024]:
                img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
                if size <= 512:
                    img.resize((size * 2, size * 2), Image.LANCZOS).save(
                        iconset / f"icon_{size}x{size}@2x.png"
                    )
            icns_path = OUTPUT_DIR / "AppIcon.icns"
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)], check=True)
            print(f"ICNS: {icns_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"iconutil failed ({e})")


if __name__ == "__main__":
    generate_icon()
