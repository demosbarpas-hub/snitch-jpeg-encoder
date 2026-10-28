// Copyright 2026 Dimosthenis Karamparpas
// SPDX-License-Identifier: Apache-2.0
//
// Baseline JPEG encoder on one Snitch compute core, fed by the DM core.
//
// The DM core streams one 16x16 RGB tile (one 4:2:0 MCU) at a time from
// `image` (L3) into TCDM with a 2-D DMA transfer. Compute core 0 runs the
// six pipeline stages on the tile and appends the Huffman-coded bits to an
// L1 payload buffer, which the DM core flushes to `jpeg_out` (L3) after
// every MCU. On exit, `jpeg_out[0 .. jpeg_size)` holds a complete JFIF file.

#include <stdint.h>

#include "data.h"
#include "jpeg.h"
#include "snrt.h"

#define PAYLOAD_BYTES 4096

static void write_jpeg_headers(BitWriter *w, uint16_t width, uint16_t height) {
    // SOI + APP0 (JFIF)
    static const uint8_t soi_app0[20] = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
        0x00, 0x01, 0x01, 0x00, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00};
    for (int i = 0; i < 20; i++) write_byte_out(w, soi_app0[i]);

    // DQT: luma (table 0) and chroma (table 1), in zig-zag order
    write_byte_out(w, 0xFF);
    write_byte_out(w, 0xDB);
    write_byte_out(w, 0x00);
    write_byte_out(w, 0x43);
    write_byte_out(w, 0x00);
    for (int i = 0; i < 64; i++)
        write_byte_out(w, (uint8_t)luma_quantizer[zigzag_map[i]]);
    write_byte_out(w, 0xFF);
    write_byte_out(w, 0xDB);
    write_byte_out(w, 0x00);
    write_byte_out(w, 0x43);
    write_byte_out(w, 0x01);
    for (int i = 0; i < 64; i++)
        write_byte_out(w, (uint8_t)chroma_quantizer[zigzag_map[i]]);

    // DHT: standard DC/AC tables for luma and chroma
    write_dht_marker(w, std_luma_dc_bits, std_luma_dc_vals, 0x0C, 0x00);
    write_dht_marker(w, luma_ac_bits, luma_ac_vals, 0xA2, 0x10);
    write_dht_marker(w, std_chroma_dc_bits, std_chroma_dc_vals, 0x0C, 0x01);
    write_dht_marker(w, chroma_ac_bits, chroma_ac_vals, 0xA2, 0x11);

    // SOF0: 8-bit, 3 components, Y sampled 2x2, Cb/Cr 1x1 (4:2:0)
    write_byte_out(w, 0xFF);
    write_byte_out(w, 0xC0);
    write_byte_out(w, 0x00);
    write_byte_out(w, 0x11);
    write_byte_out(w, 0x08);
    write_byte_out(w, (height >> 8) & 0xFF);
    write_byte_out(w, height & 0xFF);
    write_byte_out(w, (width >> 8) & 0xFF);
    write_byte_out(w, width & 0xFF);
    write_byte_out(w, 0x03);
    write_byte_out(w, 0x01);
    write_byte_out(w, 0x22);
    write_byte_out(w, 0x00);
    write_byte_out(w, 0x02);
    write_byte_out(w, 0x11);
    write_byte_out(w, 0x01);
    write_byte_out(w, 0x03);
    write_byte_out(w, 0x11);
    write_byte_out(w, 0x01);

    // SOS
    static const uint8_t sos[14] = {0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00,
                                    0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00};
    for (int i = 0; i < 14; i++) write_byte_out(w, sos[i]);
}

int main() {
    const uint16_t width = img_width;
    const uint16_t height = img_height;
    const uint32_t mcus_wide = width / MCU_W;
    const uint32_t total_mcus = mcus_wide * (height / MCU_H);
    const uint32_t core_idx = snrt_cluster_core_idx();
    const int is_worker = snrt_is_compute_core() && core_idx == 0;

    // ---------------------------------------------------------------------
    // TCDM layout (identical on all cores; only core 0 and the DM core use it)
    // ---------------------------------------------------------------------
    uintptr_t base = ALIGN_UP_TCDM((uintptr_t)snrt_l1_next());
    volatile uint32_t *payload_bytes = (volatile uint32_t *)base;
    double *lut = (double *)ALIGN_UP_TCDM(base + sizeof(uint32_t));
    double *l1_matrix =
        (double *)ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));
    double *l1_recip_lum =
        (double *)ALIGN_UP_TCDM((uintptr_t)l1_matrix + 64 * sizeof(double));
    double *l1_recip_chr =
        (double *)ALIGN_UP_TCDM((uintptr_t)l1_recip_lum + 64 * sizeof(double));
    uint8_t *input_rgb =
        (uint8_t *)ALIGN_UP_TCDM((uintptr_t)l1_recip_chr + 64 * sizeof(double));
    double *ws_a =
        (double *)ALIGN_UP_TCDM((uintptr_t)input_rgb + MCU_TOTAL_PIXELS * 3);
    double *ws_b =
        (double *)ALIGN_UP_TCDM((uintptr_t)ws_a + 384 * sizeof(double)) +
        1;  // +8 B: different bank than ws_a
    uint8_t *l1_zigzag =
        (uint8_t *)ALIGN_UP_TCDM((uintptr_t)ws_b + 512 * sizeof(double));
    uint8_t *payload = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)l1_zigzag + 64);

    // Buffer aliasing between stages (see thesis Section 5.2.3)
    double *Y = ws_a, *CbCr = ws_b;        // stage 1
    double *Cb = ws_a + MCU_TOTAL_PIXELS;  // stage 2
    double *Cr = ws_a + MCU_TOTAL_PIXELS + MCU_CHROMA_PIXELS;
    double *dct_Y = ws_b;  // stage 3
    double *dct_Cb = ws_b + MCU_TOTAL_PIXELS;
    double *dct_Cr = ws_b + MCU_TOTAL_PIXELS + MCU_CHROMA_PIXELS;
    double *q_Y = ws_a, *q_Cb = Cb, *q_Cr = Cr;  // stage 4
    int16_t *zz_Y = (int16_t *)ws_b;             // stage 5
    int16_t *zz_Cb = zz_Y + MCU_TOTAL_PIXELS;
    int16_t *zz_Cr = zz_Y + MCU_TOTAL_PIXELS + MCU_CHROMA_PIXELS;

    BitWriter writer;
    int prev_dc_Y = 0, prev_dc_Cb = 0, prev_dc_Cr = 0;
    uint32_t out_pos = 0;  // DM core: bytes written to jpeg_out
    uint32_t cyc[6] = {0};
    uint32_t t0;

    // Constant tables -> L1
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1_matrix, (void *)dct_m_l2, 64 * sizeof(double));
        snrt_dma_start_1d(l1_recip_lum, (void *)quant_recip_lum_l2,
                          64 * sizeof(double));
        snrt_dma_start_1d(l1_recip_chr, (void *)quant_recip_chr_l2,
                          64 * sizeof(double));
        snrt_dma_wait_all();
    }
    // Initialisation and JFIF headers
    if (is_worker) {
        init_standard_ac_tables();
        for (int i = 0; i < 256; i++) lut[i] = (double)i;
        for (int i = 0; i < 64; i++) l1_zigzag[i] = zigzag_map[i];
        init_bit_writer(&writer, payload);
        write_jpeg_headers(&writer, width, height);
        *payload_bytes = writer.byte_pos;
    }
    snrt_cluster_hw_barrier();

    // Flush headers to L3
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(jpeg_out, payload, *payload_bytes);
        snrt_dma_wait_all();
        out_pos = *payload_bytes;
    }
    snrt_cluster_hw_barrier();
    if (is_worker) writer.byte_pos = 0;

    // ---------------------------------------------------------------------
    // MCU streaming loop
    // ---------------------------------------------------------------------
    for (uint32_t mcu = 0; mcu < total_mcus; mcu++) {
        if (snrt_is_dm_core()) {
            uint32_t row = mcu / mcus_wide, col = mcu % mcus_wide;
            const uint8_t *src =
                &image[(row * MCU_H * width + col * MCU_W) * 3];
            snrt_dma_start_2d(input_rgb, (void *)src, MCU_W * 3, MCU_W * 3,
                              width * 3, MCU_H);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (is_worker) {
            t0 = snrt_mcycle();
            opt_YCBCR_conversion(input_rgb, Y, CbCr, lut, MCU_TOTAL_PIXELS);
            cyc[0] += snrt_mcycle() - t0;

            t0 = snrt_mcycle();
            opt_chroma_420_2d_ssr(CbCr, Cb, Cr, MCU_W, MCU_H);
            cyc[1] += snrt_mcycle() - t0;

            t0 = snrt_mcycle();
            for (int b = 0; b < 4; b++)
                opt_dct_forward(&Y[b * 64], &dct_Y[b * 64], l1_matrix);
            opt_dct_forward(Cb, dct_Cb, l1_matrix);
            opt_dct_forward(Cr, dct_Cr, l1_matrix);
            cyc[2] += snrt_mcycle() - t0;

            t0 = snrt_mcycle();
            for (int b = 0; b < 4; b++)
                opt_quant(&dct_Y[b * 64], &q_Y[b * 64], l1_recip_lum);
            opt_quant(dct_Cb, q_Cb, l1_recip_chr);
            opt_quant(dct_Cr, q_Cr, l1_recip_chr);
            cyc[3] += snrt_mcycle() - t0;

            t0 = snrt_mcycle();
            opt_zigzag_and_diff(q_Y, zz_Y, &prev_dc_Y, 4, l1_zigzag);
            opt_zigzag_and_diff(q_Cb, zz_Cb, &prev_dc_Cb, 1, l1_zigzag);
            opt_zigzag_and_diff(q_Cr, zz_Cr, &prev_dc_Cr, 1, l1_zigzag);
            cyc[4] += snrt_mcycle() - t0;

            t0 = snrt_mcycle();
            for (int b = 0; b < 4; b++)
                naive_huffman_rle_encode_block(&zz_Y[b * 64], &writer, 0);
            naive_huffman_rle_encode_block(zz_Cb, &writer, 1);
            naive_huffman_rle_encode_block(zz_Cr, &writer, 1);
            cyc[5] += snrt_mcycle() - t0;

            *payload_bytes = writer.byte_pos;
        }
        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core() && *payload_bytes > 0) {
            snrt_dma_start_1d(&jpeg_out[out_pos], payload, *payload_bytes);
            snrt_dma_wait_all();
            out_pos += *payload_bytes;
        }
        snrt_cluster_hw_barrier();
        if (is_worker)
            writer.byte_pos = 0;  // partial byte stays in the bit buffer
    }

    // ---------------------------------------------------------------------
    // Pad the last byte, append EOI and flush the tail
    // ---------------------------------------------------------------------
    if (is_worker) {
        flush_bit_writer(&writer);
        write_byte_out(&writer, 0xFF);
        write_byte_out(&writer, 0xD9);
        *payload_bytes = writer.byte_pos;
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(&jpeg_out[out_pos], payload, *payload_bytes);
        snrt_dma_wait_all();
        out_pos += *payload_bytes;
        jpeg_size = out_pos;
    }
    snrt_cluster_hw_barrier();

    if (is_worker) {
        uint32_t total = 0;
        for (int i = 0; i < 6; i++) total += cyc[i];
        printf(
            "JPEG %ux%u, %u MCUs: ycbcr %u, chroma %u, dct %u, quant %u, "
            "zigzag %u, huffman %u, total %u cycles\n",
            width, height, total_mcus, cyc[0], cyc[1], cyc[2], cyc[3], cyc[4],
            cyc[5], total);
    }
    return 0;
}
