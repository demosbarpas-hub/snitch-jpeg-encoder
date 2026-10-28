# Tools

| Script | Purpose |
|---|---|
| `image_to_header.py` | Convert any image to `image_data.h` (optional `--crop X Y W H` / `--resize W H`). |
| `log_to_jpg.py` | Extract the JPEG hex dump from a simulator log and save it as `.jpg`. |
| `jpeg_reference.py` | Python model of the encoder arithmetic; `--compare run.log` checks a simulator output byte-for-byte. Matches the naive and FP64 encoders exactly. |

Requires Python 3 with `numpy` and `pillow`.
