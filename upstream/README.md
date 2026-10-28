# Contributing the encoder to pulp-platform/snitch_cluster

This folder holds a version of the FP64 encoder shaped as an upstream kernel,
following the layout of the existing `sw/kernels/misc/*` kernels:

```
sw/kernels/misc/jpeg/
├── app.mk                 build rules (datagen.mk + common.mk)
├── data/params.json       image size, seed, optional image file
├── scripts/datagen.py     generates data.h: input image + jpeg_out/jpeg_size output buffers
├── scripts/jpeg_golden.py Python golden model of the encoder arithmetic
├── scripts/verify.py      runs the simulation, checks the bitstream byte-for-byte
└── src/
    ├── jpeg.h             pipeline kernels (from apps/fp64_multicore, unchanged except the fix below)
    └── main.c             single-core orchestration: DM core streams MCUs, core 0 encodes
```

`0001-sw-Add-baseline-JPEG-encoder-kernel-misc-jpeg.patch` is the same change as a git commit
against `develop` (commit 3299c73), including the two registration lines in `make/sw.mk` and
`test/run.yaml`.

## What has been checked

| Check | Result |
|---|---|
| `datagen.py` with upstream `snitch.util.sim.data_utils` | generates `data.h` |
| Golden model vs. simulator output (512×512 Lena, naive and FP64 programs) | byte-identical, 24,104 B |
| Golden model output decodes as a valid JPEG | yes |
| `clang-format` 10 (upstream CI version), `flake8` 6.0.0 `--max-line-length 100`, `reuse lint` | pass |
| C front-end (clang, RV32IMAFD) on `main.c` + `jpeg.h` | no C errors; only Snitch-specific instructions (`frep`, `copift`) are unknown to a stock assembler |
| **Build with the Snitch LLVM toolchain and run on the RTL model** | **not done yet — do this before opening the PR** |

`main.c` is new code (single-core orchestration with `jpeg_out` / `jpeg_size` symbols instead of
the fixed DRAM address used in the thesis programs); the kernels in `jpeg.h` are the ones measured
in the thesis. One change to them: `opt_quant()` listed `ft10` both as the register holding the
magic rounding constant and as clobbered, which current clang rejects; `ft10` was removed from the
clobber list (it is only read, so behaviour is unchanged). The same fix was applied in `apps/fp64_multicore/`.

## Steps

```bash
# 1. Fork pulp-platform/snitch_cluster on GitHub, enable Actions on the fork, then:
git clone https://github.com/<you>/snitch_cluster.git && cd snitch_cluster
git checkout -b jpeg-kernel origin/develop
git am /path/to/0001-sw-Add-baseline-JPEG-encoder-kernel-misc-jpeg.patch

# 2. Build and verify on the RTL model (see snitch_cluster docs for the environment)
make -C target/snitch_cluster sw
cd target/snitch_cluster
../../sw/kernels/misc/jpeg/scripts/verify.py bin/snitch_cluster.vlt \
    ../../sw/kernels/misc/jpeg/build/jpeg.elf --save-jpeg out.jpg

# 3. Push, wait for the fork CI, open a PR to `develop`
```

The upstream CONTRIBUTING guide asks to open an issue first for larger contributions.
Suggested text:

> **Proposal: baseline JPEG encoder as an application kernel**
>
> As part of my diploma thesis at the University of Patras I implemented a complete baseline
> JPEG encoder (4:2:0, standard tables) for the Snitch cluster, using SSR/ISSR streams, FREP and
> the copift queues, with the DM core streaming MCUs through TCDM. On a single core it encodes a
> 16×16 MCU in ~33k cycles (3.7× over a scalar C version). I would like to contribute it as
> `sw/kernels/misc/jpeg` with a datagen script and a verify script that checks the bitstream
> byte-for-byte against a Python golden model. Would this be of interest, and is `misc/` the
> right place?

## Suggested PR description

> **sw: Add baseline JPEG encoder kernel (misc/jpeg)**
>
> Adds a single-core baseline JPEG encoder (RGB→YCbCr, 4:2:0 subsampling, 8×8 DCT,
> quantization, zig-zag/DPCM, Huffman with standard tables) as an application kernel.
>
> - Kernels use SSR/ISSR (4-D patterns for transpose-free DCT, indexed zig-zag), FREP,
>   magic-number rounding and the copift FPU→integer queue.
> - The DM core streams one MCU at a time with 2-D DMA and flushes the bitstream after each MCU.
> - `datagen.py` builds a deterministic synthetic image (or loads one); `verify.py` checks the
>   output byte-for-byte against a Python golden model.
> - Registered in `make/sw.mk` and `test/run.yaml`.
>
> Thesis measurements of these kernels (Verilator, 32×32 image): ~33,129 cycles per MCU on one core
> (YCbCr 4,040 · chroma 1,456 · DCT 9,669 · quant 2,115 · zig-zag 1,035 · Huffman 14,815).
