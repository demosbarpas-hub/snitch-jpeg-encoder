#!/usr/bin/env python3
# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0
"""Generate the input image and output buffers for the JPEG kernel."""

import numpy as np

import snitch.util.sim.data_utils as du

# AXI splits bursts crossing 4KB address boundaries
BURST_ALIGNMENT = 4096


def synthetic_image(width, height, seed):
    """Deterministic RGB test image with smooth regions, edges and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 128 + 100 * np.sin(2 * np.pi * x / width)
    g = 255 * y / max(height - 1, 1)
    b = 64 + 128 * ((x + y) / (width + height))
    disc = (x - width / 2) ** 2 + (y - height / 2) ** 2 < (min(width, height) / 4) ** 2
    r[disc], g[disc], b[disc] = 230, 40, 40
    img = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (height, width, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def load_image(path, width, height):
    from PIL import Image
    img = Image.open(path).convert('RGB').resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


class JpegDataGen(du.DataGen):

    def emit_header(self, **kwargs):
        header = [super().emit_header()]
        width, height = kwargs['width'], kwargs['height']
        assert width % 16 == 0 and height % 16 == 0, 'width and height must be multiples of 16'
        if kwargs.get('image'):
            image = load_image(kwargs['image'], width, height)
        else:
            image = synthetic_image(width, height, kwargs['seed'])
        # Upper bound of the encoded size: raw size plus headers
        max_out = width * height * 3 + 1024

        header += [du.format_scalar_definition('uint32_t', 'img_width', width)]
        header += [du.format_scalar_definition('uint32_t', 'img_height', height)]
        header += [du.format_array_definition('uint8_t', 'image', image.flatten(),
                   alignment=BURST_ALIGNMENT, section=kwargs['section'])]
        header += [du.format_array_declaration('uint8_t', 'jpeg_out', [max_out],
                   alignment=BURST_ALIGNMENT, section=kwargs['section'])]
        header += [du.format_scalar_declaration('uint32_t', 'jpeg_size')]
        return '\n\n'.join(header)


if __name__ == '__main__':
    JpegDataGen().main()
