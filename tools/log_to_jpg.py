#!/usr/bin/env python3
# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0
"""Extract the JPEG hex dump printed by the Snitch encoders and save it as a .jpg.

Usage: log_to_jpg.py simulator_output.log out.jpg
"""
import re
import sys

if len(sys.argv) != 3:
    sys.exit(__doc__)
text = open(sys.argv[1], errors="replace").read()
m = re.search(r"START JPEG HEX DUMP[^\n]*\n(.*?)--- END JPEG HEX DUMP", text, re.S)
if not m:
    sys.exit("no 'START JPEG HEX DUMP ... END JPEG HEX DUMP' block found")
data = bytes.fromhex(re.sub(r"[^0-9A-Fa-f]", "", m.group(1)))
open(sys.argv[2], "wb").write(data)
print(f"wrote {len(data)} bytes to {sys.argv[2]}")
