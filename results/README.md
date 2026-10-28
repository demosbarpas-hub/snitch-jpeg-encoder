# Results

## `lena_32/` — the thesis test image (32×32 crop of Lena at (220, 220), 4 MCUs)

| File | Encoder | Size |
|---|---|---:|
| `naive.jpg` | naive | 817 B |
| `fp64.jpg` | FP64 | 817 B (identical to naive) |
| `fp16.jpg` | FP16 SIMD / hybrid | 822 B |

## `lena_512/` — full 512×512 image (1,024 MCUs)

Earlier full-image runs, made before the final optimizations
(the FP64 run used an earlier version that still had the scalar zig-zag stage,
and the FP16 run an earlier version of `apps/fp16_simd`).
Logs contain per-stage totals and the full hex dump.

| Stage (total cycles) | Naive v19 | FP64 plus_4 | FP16 v8 |
|---|---:|---:|---:|
| YCbCr | 14,851,072 | 4,131,840 | 2,151,424 |
| Chroma 4:2:0 | 2,354,190 | 1,495,040 | 743,424 |
| 2D-DCT | 41,841,677 | 8,735,784 | 6,957,116 |
| Quantization | 41,359,363 | 7,153,664 | 477,181 |
| Zig-zag + diff | 9,610,294 | 9,597,020 | 23,742,416 |
| Huffman | 9,787,206 | 9,701,465 | 9,747,018 |
| **Total** | **119,803,802** | **40,814,813** | **43,818,579** |
| Output size | 24,104 B | 24,104 B | 24,212 B |

`tools/jpeg_reference.py lena_color_512.png --compare lena_512/naive_v19.log` → byte-identical.
