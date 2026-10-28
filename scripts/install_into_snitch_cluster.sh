#!/usr/bin/env bash
# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0
#
# Copy the JPEG encoders into a snitch_cluster checkout as software tests,
# so they are built by `make sw` like any file in sw/tests/src.
#
# Usage: scripts/install_into_snitch_cluster.sh /path/to/snitch_cluster
set -euo pipefail
SN_ROOT=${1:?usage: $0 /path/to/snitch_cluster}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DST="$SN_ROOT/sw/tests/src"
[ -d "$DST" ] || { echo "error: $DST not found (is $SN_ROOT a snitch_cluster checkout?)" >&2; exit 1; }
cp "$HERE/apps/common/image_data.h" "$DST/"
for f in "$HERE"/apps/*/*.c; do cp "$f" "$DST/"; echo "installed $(basename "$f")"; done
echo
echo "Build:  make -C \"$SN_ROOT/target/snitch_cluster\" sw"
echo "Run:    cd \"$SN_ROOT/target/snitch_cluster\" && bin/snitch_cluster.vlt sw/tests/build/jpeg_hybrid_fp16_fp64.elf | tee run.log"
echo "Image:  $HERE/tools/log_to_jpg.py run.log out.jpg"
