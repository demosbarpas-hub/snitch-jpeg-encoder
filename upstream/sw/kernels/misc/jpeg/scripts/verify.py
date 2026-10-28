#!/usr/bin/env python3
# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0
"""Run the JPEG kernel and compare its bitstream with the golden model."""

import sys

import numpy as np

from jpeg_golden import encode
from snitch.util.sim.verif_utils import Verifier


class JpegVerifier(Verifier):

    OUTPUT_UIDS = ['jpeg_out', 'jpeg_size']

    def parser(self):
        p = super().parser()
        p.add_argument('--save-jpeg', help='Also write the produced JPEG to this file')
        return p

    def get_actual_results(self):
        size = int(self.get_output_from_symbol('jpeg_size', 'uint32_t')[0])
        data = self.get_output_from_symbol('jpeg_out', 'uint8_t')[:size]
        if self.args.save_jpeg:
            open(self.args.save_jpeg, 'wb').write(bytes(data))
        return np.asarray(data, dtype=np.uint8)

    def get_expected_results(self):
        width = int(self.get_input_from_symbol('img_width', 'uint32_t')[0])
        height = int(self.get_input_from_symbol('img_height', 'uint32_t')[0])
        image = self.get_input_from_symbol('image', 'uint8_t').reshape(height, width, 3)
        return np.frombuffer(encode(image), dtype=np.uint8)

    def check_results(self, actual, expected):
        if len(actual) != len(expected):
            print(f'JPEG size mismatch: got {len(actual)} bytes, expected {len(expected)}')
            return 1
        mismatches = np.flatnonzero(actual != expected)
        if mismatches.size:
            print(f'{mismatches.size} bytes differ, first at offset {mismatches[0]}')
            return 1
        print(f'JPEG bitstream matches the golden model ({len(actual)} bytes)')
        return 0


if __name__ == '__main__':
    sys.exit(JpegVerifier().main())
