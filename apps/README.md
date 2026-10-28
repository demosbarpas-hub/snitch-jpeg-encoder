# Final encoders

Each `.c` file is a complete, self-contained program for the snitch_cluster runtime.
All of them read the image from [`common/image_data.h`](common/image_data.h)
(`IMG_WIDTH`, `IMG_HEIGHT`, `lena_pixels[]`, RGB row-major; default: 32×32 crop of Lena at (220, 220))
and write a JFIF file to DRAM at `0x80208000`, then print per-stage cycle counts and a hex dump.

| File | Thesis | Notes |
|---|---|---|
| `naive/jpeg_naive.c` | §4.1, Table 4.3 | Scalar C reference. DM core streams one MCU at a time via 2-D DMA. |
| `fp64_multicore/jpeg_fp64_multicore.c` | Ch. 5, Table 5.1 | Final FP64 pipeline on all 8 compute cores: TCDM root directory, per-core 14,080 B workspaces with memory aliasing, dual-level bank staggering (`core_bank_offset = 32·id`, +16/+0/+8 B intra-core), Huffman prefetching. Setting `core_bank_offset` and the +16/+8 B intra-core offsets to 0 gives the no-staggering control of §5.4. See the note in the top-level README on how tiles are distributed. |
| `fp16_simd/jpeg_fp16_simd.c` | §4.2–4.6, Table 4.3 | Packed-SIMD FP16 pipeline with the original software unpack in the zig-zag stage. Quality factor configurable with `JPEG_QUALITY`. |
| `hybrid_fp16_fp64/jpeg_hybrid_fp16_fp64.c` | §4.8, Table 4.5 | FP16 stages 1–4 + `flh`/`fcvt.d.h` widening + FP64 ISSR/copift zig-zag. `USE_HYBRID_ZIGZAG 0` falls back to the FP16 unpack for A/B comparison (identical output). Fastest single-core version. |

Build with `../scripts/install_into_snitch_cluster.sh`, see the top-level README.
