#!/usr/bin/env python3
# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0
"""Convert an image into the image_data.h header used by the Snitch encoders.

The image is converted to RGB, optionally cropped/resized, and written as a
row-major R,G,B byte array named `lena_pixels` together with IMG_WIDTH and
IMG_HEIGHT. Width and height must be multiples of 16 (one 4:2:0 MCU).

Examples:
  image_to_header.py lena_color_512.png -o apps/common/image_data.h
  image_to_header.py lena_color_512.png --crop 240 240 32 32 -o image_data.h
  image_to_header.py photo.jpg --resize 64 64 -o image_data.h
"""
import argparse

import numpy as np
from PIL import Image

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("image")
p.add_argument("-o", "--output", default="image_data.h")
p.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
p.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"))
a = p.parse_args()

img = Image.open(a.image).convert("RGB")
if a.crop:
    x, y, w, h = a.crop
    img = img.crop((x, y, x + w, y + h))
if a.resize:
    img = img.resize(tuple(a.resize), Image.LANCZOS)
w, h = img.size
if w % 16 or h % 16:
    p.error(f"image is {w}x{h}; width and height must be multiples of 16")

px = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
with open(a.output, "w", newline="\n") as f:
    f.write("#ifndef IMAGE_DATA_H\n#define IMAGE_DATA_H\n\n#include <stdint.h>\n\n")
    f.write(f"#define IMG_WIDTH {w}\n#define IMG_HEIGHT {h}\n\n")
    f.write("const uint8_t lena_pixels[] = {\n")
    for r, g, b in px:
        f.write(f"    {r}, {g}, {b},\n")
    f.write("};\n\n#endif // IMAGE_DATA_H\n")
print(f"wrote {a.output}: {w}x{h} ({w * h * 3} bytes, {(w // 16) * (h // 16)} MCUs)")
