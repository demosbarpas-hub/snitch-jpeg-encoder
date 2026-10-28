#!/usr/bin/env python3
# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0
"""Golden model of the Snitch JPEG encoder (bit-exact with src/main.c).

Mirrors the arithmetic of apps/naive/jpeg_naive.c (FP64, standard
quality-50 tables, 4:2:0, baseline Huffman) so that its output can be
compared byte-for-byte with the hex dump printed by the simulator.

Usage:
  jpeg_reference.py IMAGE.png -o out.jpg              # encode an image
  jpeg_reference.py IMAGE.png --compare sim.log       # compare with a simulator log
  jpeg_reference.py --header apps/common/image_data.h -o out.jpg
"""
import argparse
import re
import sys

import numpy as np

DCT_M = np.array([
    [0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553],
    [0.490393, 0.415735, 0.277785, 0.097545, -0.097545, -0.277785, -0.415735, -0.490393],
    [0.461940, 0.191342, -0.191342, -0.461940, -0.461940, -0.191342, 0.191342, 0.461940],
    [0.415735, -0.097545, -0.490393, -0.277785, 0.277785, 0.490393, 0.097545, -0.415735],
    [0.353553, -0.353553, -0.353553, 0.353553, 0.353553, -0.353553, -0.353553, 0.353553],
    [0.277785, -0.490393, 0.097545, 0.415735, -0.415735, -0.097545, 0.490393, -0.277785],
    [0.191342, -0.461940, 0.461940, -0.191342, -0.191342, 0.461940, -0.461940, 0.191342],
    [0.097545, -0.277785, 0.415735, -0.490393, 0.490393, -0.415735, 0.277785, -0.097545]])

RECIP_LUM = np.array([
    0.062500, 0.090909, 0.100000, 0.062500, 0.041667, 0.025000, 0.019608, 0.016393,
    0.083333, 0.083333, 0.071429, 0.052632, 0.038462, 0.018182, 0.017544, 0.014493,
    0.071429, 0.076923, 0.062500, 0.038462, 0.025000, 0.017544, 0.014493, 0.012048,
    0.071429, 0.058824, 0.045455, 0.029412, 0.018182, 0.010417, 0.011364, 0.013158,
    0.055556, 0.045455, 0.027027, 0.017857, 0.014706, 0.009009, 0.009709, 0.011236,
    0.041667, 0.028571, 0.018182, 0.015625, 0.011364, 0.009434, 0.008850, 0.010101,
    0.020408, 0.015625, 0.012500, 0.011111, 0.009174, 0.007812, 0.007752, 0.009259,
    0.013889, 0.010870, 0.010526, 0.010309, 0.009009, 0.007937, 0.008065, 0.009434])
RECIP_CHR = np.array([
    0.058824, 0.055556, 0.041667, 0.021277] + [0.010101] * 4 + [
    0.055556, 0.047619, 0.038462, 0.015152] + [0.010101] * 4 + [
    0.041667, 0.038462, 0.017857] + [0.010101] * 5 + [
    0.021277, 0.015152] + [0.010101] * 6 + [0.010101] * 32)

Q_LUM = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
         14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
         49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
Q_CHR = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
         24, 26, 56, 99, 99, 99, 99, 99, 47, 66] + [99] * 38

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
          12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
          35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
          58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

DC_LUM_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
DC_CHR_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
DC_VALS = list(range(12))
AC_LUM_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
AC_CHR_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
AC_LUM_VALS = bytes.fromhex(
    "01020300041105122131410613516107227114328191a1082342b1c11552d1f0"
    "2433627282090a161718191a25262728292a3435363738393a43444546474849"
    "4a535455565758595a636465666768696a737475767778797a83848586878889"
    "8a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5"
    "c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8"
    "f9fa")
AC_CHR_VALS = bytes.fromhex(
    "000102031104052131061241510761711322328108144291a1b1c109233352f0"
    "156272d10a162434e125f11718191a262728292a35363738393a434445464748"
    "494a535455565758595a636465666768696a737475767778797a828384858687"
    "88898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3"
    "c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8"
    "f9fa")


def build_codes(bits, vals):
    """Canonical Huffman code assignment (JPEG Annex C)."""
    table, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            table[vals[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return table


DC_LUM = build_codes(DC_LUM_BITS, DC_VALS)
DC_CHR = build_codes(DC_CHR_BITS, DC_VALS)
AC_LUM = build_codes(AC_LUM_BITS, AC_LUM_VALS)
AC_CHR = build_codes(AC_CHR_BITS, AC_CHR_VALS)


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def bits(self, code, length):
        for i in range(length - 1, -1, -1):
            self.acc = (self.acc << 1) | ((code >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0x00)  # byte stuffing
                self.acc, self.n = 0, 0

    def flush(self):
        if self.n:
            pad = 8 - self.n
            self.bits((1 << pad) - 1, pad)  # pad with ones


def category(v):
    if v == 0:
        return 0, 0
    cat = int(abs(v)).bit_length()
    return cat, (v + (1 << cat) - 1) if v < 0 else v


def encode_block(bw, zz, dc_tab, ac_tab):
    cat, code = category(zz[0])
    bw.bits(*dc_tab[cat])
    bw.bits(code, cat)
    run = 0
    for v in zz[1:]:
        if v == 0:
            run += 1
            continue
        while run > 15:
            bw.bits(*ac_tab[0xF0])
            run -= 16
        cat, code = category(v)
        bw.bits(*ac_tab[(run << 4) | cat])
        bw.bits(code, cat)
        run = 0
    if run:
        bw.bits(*ac_tab[0x00])


def dct_quant(block, recip):
    """Naive separable DCT with sequential accumulation, as in naive_dct()."""
    s = block - 128.0
    tmp = np.zeros((8, 8))
    for k in range(8):                       # tmp[r][c] = sum_k s[r][k] * M[c][k]
        tmp = tmp + s[:, k:k + 1] * DCT_M[:, k][None, :]
    out = np.zeros((8, 8))
    for k in range(8):                       # out[r][c] = sum_k M[r][k] * tmp[k][c]
        out = out + DCT_M[:, k:k + 1] * tmp[k:k + 1, :]
    return np.rint(out.reshape(64) * recip).astype(int)  # round-half-even = nearbyint


def headers(w, h):
    o = bytearray(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x48\x00\x48\x00\x00")
    for tid, q in ((0, Q_LUM), (1, Q_CHR)):
        o += bytes([0xFF, 0xDB, 0x00, 0x43, tid]) + bytes(q[z] for z in ZIGZAG)
    for bits, vals, tc in ((DC_LUM_BITS, DC_VALS, 0x00), (AC_LUM_BITS, AC_LUM_VALS, 0x10),
                           (DC_CHR_BITS, DC_VALS, 0x01), (AC_CHR_BITS, AC_CHR_VALS, 0x11)):
        n = 2 + 1 + 16 + len(vals)
        o += bytes([0xFF, 0xC4, n >> 8, n & 0xFF, tc]) + bytes(bits) + bytes(vals)
    o += bytes([0xFF, 0xC0, 0x00, 0x11, 0x08, h >> 8, h & 0xFF, w >> 8, w & 0xFF, 0x03,
                0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01])
    o += bytes([0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00])
    return o


def encode(rgb):
    """rgb: HxWx3 uint8 array, H and W multiples of 16."""
    h, w, _ = rgb.shape
    assert h % 16 == 0 and w % 16 == 0, "image dimensions must be multiples of 16"
    px = rgb.astype(np.float64)
    R, G, B = px[..., 0], px[..., 1], px[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 0.564 * (B - Y) + 128.0
    Cr = 0.713 * (R - Y) + 128.0
    bw, prev = BitWriter(), [0, 0, 0]
    for my in range(0, h, 16):
        for mx in range(0, w, 16):
            blocks = []
            for by, bx in ((0, 0), (0, 8), (8, 0), (8, 8)):
                blocks.append((0, Y[my + by:my + by + 8, mx + bx:mx + bx + 8]))
            for comp, plane in ((1, Cb), (2, Cr)):
                t = plane[my:my + 16, mx:mx + 16]
                sub = (((t[0::2, 0::2] + t[0::2, 1::2]) + t[1::2, 0::2]) + t[1::2, 1::2]) * 0.25
                blocks.append((comp, sub))
            for comp, blk in blocks:
                q = dct_quant(blk, RECIP_LUM if comp == 0 else RECIP_CHR)
                zz = [int(q[z]) for z in ZIGZAG]
                dc = zz[0]
                zz[0], prev[comp] = dc - prev[comp], dc
                if comp == 0:
                    encode_block(bw, zz, DC_LUM, AC_LUM)
                else:
                    encode_block(bw, zz, DC_CHR, AC_CHR)
    bw.flush()
    return bytes(headers(w, h) + bw.out + b"\xFF\xD9")


def load_header(path):
    s = open(path).read()
    w = int(re.search(r"IMG_WIDTH\s+(\d+)", s).group(1))
    h = int(re.search(r"IMG_HEIGHT\s+(\d+)", s).group(1))
    body = re.search(r"\{([^}]*)\}", s).group(1)
    vals = [int(x) for x in re.findall(r"\d+", body)]
    return np.array(vals, dtype=np.uint8).reshape(h, w, 3)


def hex_from_log(path):
    s = open(path, errors="replace").read()
    m = re.search(r"START JPEG HEX DUMP[^\n]*\n(.*?)--- END JPEG HEX DUMP", s, re.S)
    if not m:
        sys.exit(f"no hex dump found in {path}")
    return bytes.fromhex(re.sub(r"[^0-9A-Fa-f]", "", m.group(1)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", help="input image (PNG etc.)")
    p.add_argument("--header", help="read pixels from an image_data.h instead")
    p.add_argument("-o", "--output", help="write the reference JPEG here")
    p.add_argument("--compare", help="simulator log containing a JPEG hex dump")
    a = p.parse_args()
    if a.header:
        rgb = load_header(a.header)
    else:
        from PIL import Image
        rgb = np.array(Image.open(a.image).convert("RGB"))
    ref = encode(rgb)
    print(f"reference: {rgb.shape[1]}x{rgb.shape[0]}, {len(ref)} bytes")
    if a.output:
        open(a.output, "wb").write(ref)
    if a.compare:
        sim = hex_from_log(a.compare)
        if sim == ref:
            print(f"MATCH: simulator output is byte-identical ({len(sim)} bytes)")
            return 0
        n = min(len(sim), len(ref))
        diff = next((i for i in range(n) if sim[i] != ref[i]), n)
        print(f"DIFFER: simulator {len(sim)} bytes, reference {len(ref)} bytes, "
              f"first difference at byte {diff}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
