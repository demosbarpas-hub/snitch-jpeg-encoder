# JPEG Compression on the RISC-V Snitch Cluster

Hardware–software co-design of a baseline JPEG encoder for the
[PULP Snitch cluster](https://github.com/pulp-platform/snitch_cluster)
(8 RV32 compute cores + 1 DMA core, 128 KiB TCDM, SSR/ISSR, FREP, packed-SIMD FP16).

Diploma thesis *"JPEG Compression on Low Power RISC-V"*, Department of Electrical and Computer Engineering, University of Patras (October 2026).
Author: **Dimosthenis Karamparpas** · Supervisor: George Theodoridis ·
Full text: [`docs/thesis_karamparpas.pdf`](docs/thesis_karamparpas.pdf)

The encoder implements the full baseline JPEG pipeline on-device:
RGB→YCbCr, 4:2:0 chroma subsampling, 8×8 2D-DCT, quantization (quality 50),
zig-zag + DC differential coding, and Huffman entropy coding with standard tables,
producing a valid JFIF file in DRAM. It was developed in five stages, each kept as a
self-contained program:

| Program | Precision | Key techniques | Folder |
|---|---|---|---|
| Naive | FP64, scalar C | Reference implementation | [`apps/naive`](apps/naive) |
| FP64 (multi-core) | FP64 | SSR/ISSR streaming, FREP, 4-D address patterns for transpose-free DCT, magic-number rounding, copift FPU→integer queues, DM-core DMA streaming, dual-level TCDM bank staggering, Huffman prefetching | [`apps/fp64_multicore`](apps/fp64_multicore) |
| FP16 SIMD | packed FP16 ×4 | `vf*.h` SIMD kernels for colour conversion, subsampling, DCT, quantization | [`apps/fp16_simd`](apps/fp16_simd) |
| Hybrid | FP16 + FP64 | FP16 SIMD for stages 1–4, `flh`-based widening to FP64, FP64 ISSR/copift zig-zag | [`apps/hybrid_fp16_fp64`](apps/hybrid_fp16_fp64) |

## Results

Cycles per 4:2:0 MCU (16×16 pixels, 6 blocks), cycle-accurate RTL simulation (Verilator),
32×32 test image (4 MCUs).

### Single core (thesis Tables 4.3 and 4.5)

| Stage | Naive | FP64 | FP16 SIMD | Hybrid FP16/FP64 |
|---|---:|---:|---:|---:|
| YCbCr conversion | 14,503.00 | 4,039.50 | 2,103.00 | 2,100.00 |
| Chroma 4:2:0 | 2,303.75 | 1,456.25 | 725.00 | 725.00 |
| 2D-DCT | 40,865.75 | 9,669.25 | 6,814.25 | 6,801.00 |
| Quantization | 39,958.50 | 2,114.50 | 478.00 | 462.50 |
| Zig-zag + DC diff | 9,387.25 | 1,035.00 | 10,355.25 | 2,652.00 |
| Huffman | 14,943.50 | 14,814.50 | 14,886.75 | 14,961.75 |
| **Total** | **121,961.75** | **33,129.00** | **35,362.25** | **27,702.25** |
| Speed-up vs naive | 1.00× | 3.68× | 3.45× | **4.40×** |

The FP16 pipeline is faster in four of six stages, but its original zig-zag stage
unpacked each lane in software (manual NaN-boxing and an FPU–integer round trip per
value). The hybrid fixes this with `flh` + `fcvt.d.h` widening and reuses the FP64
ISSR/copift zig-zag kernel; it produces a bit-identical bitstream to the FP16 pipeline.
(The FP16 and hybrid columns of Table 4.5 come from the same run; FP64 and naive from Table 4.3.)

### Eight compute cores (thesis Table 5.1)

| Stage | Single core | 8-core cluster (per core) |
|---|---:|---:|
| YCbCr | 4,039.50 | 4,039.25 |
| Chroma | 1,456.25 | 1,456.25 |
| DCT | 9,669.25 | 9,642.00 |
| Quant | 2,114.50 | 2,112.75 |
| Zig-zag | 1,035.00 | 1,028.50 |
| Huffman | 14,814.50 | 14,229.00 |
| **Total** | **33,129.00** | **32,507.75** |

With all eight cores busy, each core keeps its single-core cycle count (no slowdown
from TCDM contention), and Huffman prefetching removes ~585 cycles per MCU;
8 MCUs worth of work takes 32,508 cycles instead of 8 × 33,129 = 265,032 (8.15×).

> **How the 8-core program works.** `jpeg_fp64_multicore.c` has the DM core copy the
> *same* 16×16 tile into all eight cores' private TCDM workspaces each iteration; every
> core runs the full pipeline concurrently and checks that all eight produce identical
> bitstreams, and core 0's bitstream is written to DRAM. It therefore measures per-core
> throughput under full-cluster contention and the TCDM layout, which is what Table 5.1
> reports. Distributing *different* MCUs to the cores and stitching their bitstreams
> (e.g. with JPEG restart markers) is listed as future work.

### Stall analysis (thesis §5.4, per worker core)

| Configuration | TCDM bank stalls | RAW accelerator hazards | RAW load-use hazards |
|---|---:|---:|---:|
| Baseline | 78 | 267 | 2,529 |
| + bank staggering | 78 | 268 | 2,524 |
| + Huffman prefetching | 81 | 272 | **1,033** |

### Correctness

[`tools/jpeg_reference.py`](tools/jpeg_reference.py) is a Python model of the encoder's
arithmetic. On the 512×512 Lena image it reproduces the simulator's output **byte for byte**
(24,104 bytes) for both the naive and the FP64 pipelines, and on the 32×32 test image
(817 bytes). See [`results/`](results).

## Repository layout

```
apps/          Final encoders (one self-contained .c each) + shared image header
tools/         Image → header conversion, log → .jpg, Python reference encoder
results/       Simulator logs and output images (Lena 512×512 and 32×32)
upstream/      Draft contribution for pulp-platform/snitch_cluster (sw/kernels/misc/jpeg)
docs/          Thesis PDF
```

## Building and running

The programs are written against the snitch_cluster runtime (`snrt.h`) and were built as
software tests of a snitch_cluster checkout (64-bit FPU configuration, `SNRT_SUPPORTS_FREP`).

```bash
# 1. Set up snitch_cluster as described in its documentation (toolchain + Verilator model)
git clone https://github.com/pulp-platform/snitch_cluster.git
# 2. Copy the encoders into sw/tests/src
scripts/install_into_snitch_cluster.sh /path/to/snitch_cluster
# 3. Build and simulate
cd /path/to/snitch_cluster/target/snitch_cluster
make sw
bin/snitch_cluster.vlt sw/tests/build/jpeg_hybrid_fp16_fp64.elf | tee run.log
# 4. Recover the JPEG from the hex dump and check it
tools/log_to_jpg.py run.log out.jpg
tools/jpeg_reference.py --header apps/common/image_data.h --compare run.log   # FP64/naive: exact match
```

To encode a different image, regenerate the header (dimensions must be multiples of 16):

```bash
tools/image_to_header.py lena_color_512.png --crop 220 220 32 32 -o apps/common/image_data.h  # the thesis test image
```

Output is written to DRAM at `0x80208000` and printed as a hex dump by core 0.

## Citation

```
D. Karamparpas, "JPEG Compression on Low Power RISC-V",
Diploma Thesis, Dept. of Electrical and Computer Engineering, University of Patras, Oct. 2026.
```

## License

Code: [Apache License 2.0](LICENSE) (same as snitch_cluster). The thesis PDF in `docs/` is
© the author, all rights reserved. The snitch_cluster runtime is not included; see its repository.
