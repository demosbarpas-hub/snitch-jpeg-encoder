// Copyright 2026 Dimosthenis Karamparpas
// SPDX-License-Identifier: Apache-2.0
//
// Baseline JPEG encoder kernels for Snitch (FP64, SSR/ISSR + FREP + copift).
// Stages: RGB->YCbCr, 4:2:0 subsampling, 8x8 2D-DCT, quantization (quality 50),
// zig-zag + DC DPCM, Huffman coding with the standard JPEG tables.

#pragma once

#include <stdint.h>

#include "snrt.h"

/* --- Configuration & Macros --- */
#define BANK_SIZE 8
#define NUM_BANKS 32
#define ROW_SIZE (NUM_BANKS * BANK_SIZE)

#define ALIGN_UP_TCDM(addr) ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)

/* --- Fixed dimensions of a standard 4:2:0 MCU --- */
#define MCU_W 16
#define MCU_H 16
#define MCU_TOTAL_PIXELS (MCU_W * MCU_H)          // Always 256
#define MCU_CHROMA_PIXELS (MCU_TOTAL_PIXELS / 4)  // Always 64

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define CLIP(n, min, max) MIN((MAX((n), (min))), (max))

/* --- Constant DCT & quantization tables (L2) --- */
const double dct_m_l2[8][8] __attribute__((aligned(64))) = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553,
     0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545, -0.097545, -0.277785, -0.415735,
     -0.490393},
    {0.461940, 0.191342, -0.191342, -0.461940, -0.461940, -0.191342, 0.191342,
     0.461940},
    {0.415735, -0.097545, -0.490393, -0.277785, 0.277785, 0.490393, 0.097545,
     -0.415735},
    {0.353553, -0.353553, -0.353553, 0.353553, 0.353553, -0.353553, -0.353553,
     0.353553},
    {0.277785, -0.490393, 0.097545, 0.415735, -0.415735, -0.097545, 0.490393,
     -0.277785},
    {0.191342, -0.461940, 0.461940, -0.191342, -0.191342, 0.461940, -0.461940,
     0.191342},
    {0.097545, -0.277785, 0.415735, -0.490393, 0.490393, -0.415735, 0.277785,
     -0.097545}};

const double quant_recip_lum_l2[64] __attribute__((aligned(64))) = {
    0.062500, 0.090909, 0.100000, 0.062500, 0.041667, 0.025000, 0.019608,
    0.016393, 0.083333, 0.083333, 0.071429, 0.052632, 0.038462, 0.018182,
    0.017544, 0.014493, 0.071429, 0.076923, 0.062500, 0.038462, 0.025000,
    0.017544, 0.014493, 0.012048, 0.071429, 0.058824, 0.045455, 0.029412,
    0.018182, 0.010417, 0.011364, 0.013158, 0.055556, 0.045455, 0.027027,
    0.017857, 0.014706, 0.009009, 0.009709, 0.011236, 0.041667, 0.028571,
    0.018182, 0.015625, 0.011364, 0.009434, 0.008850, 0.010101, 0.020408,
    0.015625, 0.012500, 0.011111, 0.009174, 0.007812, 0.007752, 0.009259,
    0.013889, 0.010870, 0.010526, 0.010309, 0.009009, 0.007937, 0.008065,
    0.009434};

const double quant_recip_chr_l2[64] __attribute__((aligned(64))) = {
    0.058824, 0.055556, 0.041667, 0.021277, 0.010101, 0.010101, 0.010101,
    0.010101, 0.055556, 0.047619, 0.038462, 0.015152, 0.010101, 0.010101,
    0.010101, 0.010101, 0.041667, 0.038462, 0.017857, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.021277, 0.015152, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101};

/* Quantization tables (integers, for the DQT headers) */
int luma_quantizer[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

int chroma_quantizer[64] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99,
                            99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66,
                            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

/* Standard JPEG Zig-Zag Map */
const uint8_t zigzag_map[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

// ========================================================================
// STANDARD HUFFMAN TABLES
// ========================================================================
const uint16_t std_luma_dc_code[12] = {0,  2,  3,  4,   5,   6,
                                       14, 30, 62, 126, 254, 510};
const uint8_t std_luma_dc_len[12] = {2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9};
const uint8_t std_luma_dc_bits[16] = {0, 1, 5, 1, 1, 1, 1, 1,
                                      1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t std_luma_dc_vals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

const uint16_t std_chroma_dc_code[12] = {0,  1,   2,   6,   14,   30,
                                         62, 126, 254, 510, 1022, 2046};
const uint8_t std_chroma_dc_len[12] = {2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t std_chroma_dc_bits[16] = {0, 3, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 0, 0, 0, 0, 0};
const uint8_t std_chroma_dc_vals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

uint16_t std_luma_ac_code[256];
uint8_t std_luma_ac_len[256];
uint16_t std_chroma_ac_code[256];
uint8_t std_chroma_ac_len[256];

const uint8_t luma_ac_bits[16] = {0, 2, 1, 3, 3, 2, 4, 3,
                                  5, 5, 4, 4, 0, 0, 1, 125};
const uint8_t luma_ac_vals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
    0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
    0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

const uint8_t chroma_ac_bits[16] = {0, 2, 1, 2, 4, 4, 3, 4,
                                    7, 5, 4, 4, 0, 1, 2, 119};
const uint8_t chroma_ac_vals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41,
    0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1,
    0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74,
    0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
    0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

void init_standard_ac_tables() {
    int code = 0, k = 0;
    for (int len = 1; len <= 16; len++) {
        for (int i = 0; i < luma_ac_bits[len - 1]; i++) {
            uint8_t sym = luma_ac_vals[k++];
            std_luma_ac_code[sym] = code++;
            std_luma_ac_len[sym] = len;
        }
        code <<= 1;
    }
    code = 0;
    k = 0;
    for (int len = 1; len <= 16; len++) {
        for (int i = 0; i < chroma_ac_bits[len - 1]; i++) {
            uint8_t sym = chroma_ac_vals[k++];
            std_chroma_ac_code[sym] = code++;
            std_chroma_ac_len[sym] = len;
        }
        code <<= 1;
    }
}

/* --- Bit-writer (32-bit safe, 8-bit accumulator) --- */
typedef struct {
    uint8_t *jpeg_output;
    uint32_t byte_pos;
    uint8_t bit_buffer;
    int32_t num_bits;
} BitWriter;

void init_bit_writer(BitWriter *writer, uint8_t *target_payload) {
    writer->jpeg_output = target_payload;
    writer->byte_pos = 0;
    writer->bit_buffer = 0;
    writer->num_bits = 0;
}

void write_byte_out(BitWriter *w, uint8_t val) {
    w->jpeg_output[w->byte_pos++] = val;
}

void write_bits(BitWriter *writer, uint32_t code, int bits) {
    if (bits == 0) return;
    code &= (1U << bits) - 1;

    for (int i = bits - 1; i >= 0; i--) {
        int bit = (code >> i) & 1;
        writer->bit_buffer = (writer->bit_buffer << 1) | bit;
        writer->num_bits++;

        if (writer->num_bits == 8) {
            uint8_t byte = writer->bit_buffer;
            writer->jpeg_output[writer->byte_pos++] = byte;
            if (byte == 0xFF) {
                writer->jpeg_output[writer->byte_pos++] = 0x00;
            }
            writer->bit_buffer = 0;
            writer->num_bits = 0;
        }
    }
}

void flush_bit_writer(BitWriter *writer) {
    if (writer->num_bits > 0) {
        int missing_bits = 8 - writer->num_bits;
        uint8_t padded_byte =
            (writer->bit_buffer << missing_bits) | ((1 << missing_bits) - 1);
        writer->jpeg_output[writer->byte_pos++] = padded_byte;
        if (padded_byte == 0xFF) {
            writer->jpeg_output[writer->byte_pos++] = 0x00;
        }
        writer->bit_buffer = 0;
        writer->num_bits = 0;
    }
}

void get_huffman_category(int val, int *category, int *code) {
    if (val == 0) {
        *category = 0;
        *code = 0;
        return;
    }
    int abs_val = (val < 0) ? -val : val;
    int cat = 0;
    while (abs_val > 0) {
        cat++;
        abs_val >>= 1;
    }
    *category = cat;
    if (val < 0)
        *code = val + (1 << cat) - 1;
    else
        *code = val;
}

void write_dht_marker(BitWriter *w, const uint8_t bits[], const uint8_t vals[],
                      int num_vals, int tc_th) {
    int length = 2 + 1 + 16 + num_vals;
    write_byte_out(w, 0xFF);
    write_byte_out(w, 0xC4);
    write_byte_out(w, (length >> 8) & 0xFF);
    write_byte_out(w, length & 0xFF);
    write_byte_out(w, tc_th);
    for (int i = 0; i < 16; i++) write_byte_out(w, bits[i]);
    for (int i = 0; i < num_vals; i++) write_byte_out(w, vals[i]);
}

void opt_zigzag_and_diff(const double *in_blocks, int16_t *out_blocks,
                         int *prev_dc, int num_blocks,
                         const uint8_t *zigzag_map) {
    for (int b = 0; b < num_blocks; b++) {
        const double *in = &in_blocks[b * 64];
        int16_t *out = &out_blocks[b * 64];

        // Configure Indexed SSR: Address Generation Unit (AGU) automatically fetches
        // `in[zigzag_map[0]]`, `in[zigzag_map[1]]`... directly into the FPU
        snrt_issr_read(SNRT_SSR_DM0, (void *)in, (void *)zigzag_map, 64,
                       SNRT_SSR_IDXSIZE_U8);
        snrt_ssr_enable();

        int32_t val;

        // Unleash the FPU to convert all 64 floats to ints and push them into the FIFO queue
        asm volatile(
            "mv        t6, x0          \n"  // Clear t6
            "csrsi     copift, 0x1     \n"  // Enable F2I (Float-to-Int) hardware queues
            "frep.o    %[n], 1, 0, 0   \n"  // Unroll FPU operation 64 times
            "fcvt.w.d  t6, ft0         \n"  // FPU: pop SSR(ft0), cast to int, push to queue(t6)
            "mv        %[val], t6      \n"  // CPU: wait and pop the first value (DC)
            : [ val ] "=r"(val)
            : [ n ] "r"(63)
            : "t6", "ft0", "memory");

        // 1. Process DC component (ALU doesn't wait for memory, it pops from queue)
        int current_dc = val;
        out[0] = (int16_t)(current_dc - *prev_dc);
        *prev_dc = current_dc;

        // 2. Process AC components
        for (int i = 1; i < 64; i++) {
            asm volatile("mv %[val], t6 \n"
                         : [ val ] "=r"(val)
                         :
                         : "t6");  // Pop from F2I queue
            out[i] = (int16_t)val;
        }

        asm volatile("csrci copift, 0x1 \n" ::: "memory");  // Disable queues
        snrt_fpu_fence();
        snrt_ssr_disable();
    }
}

void naive_huffman_rle_encode_block(int16_t *zigzag_block, BitWriter *writer,
                                    int is_chroma) {
    int category, v_code;

    get_huffman_category(zigzag_block[0], &category, &v_code);
    if (!is_chroma)
        write_bits(writer, std_luma_dc_code[category],
                   std_luma_dc_len[category]);
    else
        write_bits(writer, std_chroma_dc_code[category],
                   std_chroma_dc_len[category]);
    write_bits(writer, v_code, category);

    int r = 0;
    int16_t cur = zigzag_block[1];  // prime the pipeline
    for (int i = 1; i < 64; i++) {
        int16_t next =
            (i + 1 < 64) ? zigzag_block[i + 1] : 0;  // issue next load early

        if (cur == 0) {
            r++;
        } else {
            while (r > 15) {
                if (!is_chroma)
                    write_bits(writer, std_luma_ac_code[0xF0],
                               std_luma_ac_len[0xF0]);
                else
                    write_bits(writer, std_chroma_ac_code[0xF0],
                               std_chroma_ac_len[0xF0]);
                r -= 16;
            }
            get_huffman_category(cur, &category, &v_code);
            uint8_t rs_sym = (r << 4) | category;
            if (!is_chroma)
                write_bits(writer, std_luma_ac_code[rs_sym],
                           std_luma_ac_len[rs_sym]);
            else
                write_bits(writer, std_chroma_ac_code[rs_sym],
                           std_chroma_ac_len[rs_sym]);
            write_bits(writer, v_code, category);
            r = 0;
        }
        cur = next;
    }
    if (r > 0) {
        if (!is_chroma)
            write_bits(writer, std_luma_ac_code[0x00], std_luma_ac_len[0x00]);
        else
            write_bits(writer, std_chroma_ac_code[0x00],
                       std_chroma_ac_len[0x00]);
    }
}

// ========================================================================
// OPTIMIZED ASSEMBLY + SSR PIPELINE STAGES
// ========================================================================
void opt_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr,
                          double *lut, int total_pixels) {
    register double r_c_r asm("ft3") = 0.299;
    register double r_c_g asm("ft4") = 0.587;
    register double r_c_b asm("ft5") = 0.114;
    register double r_c_cb asm("ft6") = 0.564;
    register double r_c_cr asm("ft7") = 0.713;

    snrt_issr_read(SNRT_SSR_DM0, lut, input_idx, total_pixels * 3,
                   SNRT_SSR_IDXSIZE_U8);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 8, 2, 8, 2, 8, 8 * 8 * 8, 8 * 8, 1024);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_4D, out_Y);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, total_pixels * 2, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_CbCr);
    snrt_ssr_enable();

    asm volatile(
        "frep.o %[n], 22, 0, 0            \n"
        "fmv.d ft8,  ft0                  \n"
        "fmv.d ft9,  ft0                  \n"
        "fmv.d ft10, ft0                  \n"
        "fmv.d ft11, ft0                  \n"
        "fmv.d f8,   ft0                  \n"
        "fmv.d f9,   ft0                  \n"
        "fmul.d f10, %[c_r], ft8          \n"
        "fmul.d f11, %[c_r], ft11         \n"
        "fmadd.d f10, %[c_g], ft9,  f10   \n"
        "fmadd.d f11, %[c_g], f8,   f11   \n"
        "fmadd.d f10, %[c_b], ft10, f10   \n"
        "fmadd.d f11, %[c_b], f9,   f11   \n"
        "fmv.d ft1, f10                   \n"
        "fmv.d ft1, f11                   \n"
        "fsub.d ft10, ft10, f10           \n"
        "fsub.d f9,   f9,   f11           \n"
        "fsub.d ft8,  ft8,  f10           \n"
        "fsub.d ft11, ft11, f11           \n"
        "fmul.d ft2, %[c_cb], ft10        \n"
        "fmul.d ft2, %[c_cr], ft8         \n"
        "fmul.d ft2, %[c_cb], f9          \n"
        "fmul.d ft2, %[c_cr], ft11        \n"
        :
        : [ n ] "r"(total_pixels / 2 - 1), [ c_r ] "f"(r_c_r),
          [ c_g ] "f"(r_c_g), [ c_b ] "f"(r_c_b), [ c_cb ] "f"(r_c_cb),
          [ c_cr ] "f"(r_c_cr)
        : "ft0", "ft1", "ft2", "ft8", "ft9", "ft10", "ft11", "f8", "f9", "f10",
          "f11", "memory");

    snrt_fpu_fence();
    snrt_ssr_disable();
}

void opt_chroma_420_2d_ssr(double *in_cbcr, double *out_cb_f, double *out_cr_f,
                           int w, int h) {
    register double quarter asm("ft11") = 0.25;
    double c_128 = 128.0;

    snrt_ssr_loop_4d(SNRT_SSR_DM0, 4, 2, 8, 8, 8, 32 * 8, 32, 32 * 8 * 2);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, in_cbcr);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, (w / 2) * (h / 2), sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out_cb_f);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, (w / 2) * (h / 2), sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_cr_f);

    snrt_ssr_enable();

#ifdef SNRT_SUPPORTS_FREP
    asm volatile(
        "frep.o %[total_blocks], 16, 0, 0 \n"
        "fmv.d  ft8,  ft0 \n"
        "fmv.d  ft9,  ft0 \n"
        "fmv.d  ft3,  ft0 \n"
        "fmv.d  ft10, ft0 \n"
        "fmv.d  ft4,  ft0 \n"
        "fmv.d  ft7,  ft0 \n"
        "fmv.d  ft5,  ft0 \n"
        "fmv.d  ft6,  ft0 \n"
        "fadd.d ft8, ft8, ft3 \n"
        "fadd.d ft9, ft9, ft10 \n"
        "fadd.d ft4, ft4, ft5 \n"
        "fadd.d ft7, ft7, ft6 \n"
        "fadd.d ft8, ft8, ft4 \n"
        "fadd.d ft9, ft9, ft7 \n"
        "fmadd.d ft1, ft8, %[q], %[c_128] \n"
        "fmadd.d ft2, ft9, %[q], %[c_128] \n"
        :
        : [ total_blocks ] "r"((w / 2 * h / 2) - 1), [ q ] "f"(quarter),
          [ c_128 ] "f"(c_128)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9",
          "ft10", "memory");
#endif

    snrt_fpu_fence();
    snrt_ssr_disable();
}

void opt_dct_forward(double *in, double *out, const double *dct_m) {
    register double offset asm("ft3") = -128;

    // STAGE 0: Level Shift
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 64, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void *)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void *)out);
    snrt_ssr_enable();
    asm volatile(
        "frep.o %[n], 1, 0, 0 \n"
        "fadd.d ft1, ft0, %[offset] \n"
        :
        : [ n ] "r"(63), [ offset ] "f"(offset)
        : "ft0", "ft1", "memory");
    snrt_fpu_fence();
    snrt_ssr_disable();

    // STAGE 1: Pass 1 - Rows
    snrt_ssr_loop_4d(SNRT_SSR_DM0, 8, 8, 1, 8, 0, 8, 0, 64);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void *)out);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 8, 8, 1, 8, 64, 8, 128, 0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void *)dct_m);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void *)in);
    snrt_ssr_enable();
    asm volatile(
        "frep.o %[cnts], 24, 0, 0 \n"
        "fsub.d f3, f3, f3 \n"
        "fsub.d f4, f4, f4 \n"
        "fsub.d f5, f5, f5 \n"
        "fsub.d f6, f6, f6 \n"
        "fsub.d f7, f7, f7 \n"
        "fsub.d f28, f28, f28 \n"
        "fsub.d f29, f29, f29 \n"
        "fsub.d f30, f30, f30 \n"
        "frep.o %[cnt], 8, 0, 0 \n"
        "fmadd.d f3, f0, f1, f3 \n"
        "fmadd.d f4, f0, f1, f4 \n"
        "fmadd.d f5, f0, f1, f5 \n"
        "fmadd.d f6, f0, f1, f6 \n"
        "fmadd.d f7, f0, f1, f7 \n"
        "fmadd.d f28, f0, f1, f28 \n"
        "fmadd.d f29, f0, f1, f29 \n"
        "fmadd.d f30, f0, f1, f30 \n"
        "fmadd.d f2, f0, f1, f3 \n"
        "fmadd.d f2, f0, f1, f4 \n"
        "fmadd.d f2, f0, f1, f5 \n"
        "fmadd.d f2, f0, f1, f6 \n"
        "fmadd.d f2, f0, f1, f7 \n"
        "fmadd.d f2, f0, f1, f28 \n"
        "fmadd.d f2, f0, f1, f29 \n"
        "fmadd.d f2, f0, f1, f30 \n"
        :
        : [ cnt ] "r"(6), [ cnts ] "r"(7)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f28", "f29", "f30",
          "memory");
    snrt_fpu_fence();
    snrt_ssr_disable();

    // STAGE 2: Pass 2 - Columns
    snrt_ssr_loop_4d(SNRT_SSR_DM0, 8, 8, 1, 8, 0, 8, 0, 64);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void *)dct_m);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 8, 8, 1, 8, 8, 64, 16, 0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void *)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void *)out);
    snrt_ssr_enable();
    asm volatile(
        "frep.o %[cnts], 24, 0, 0 \n"
        "fsub.d f3, f3, f3 \n"
        "fsub.d f4, f4, f4 \n"
        "fsub.d f5, f5, f5 \n"
        "fsub.d f6, f6, f6 \n"
        "fsub.d f7, f7, f7 \n"
        "fsub.d f28, f28, f28 \n"
        "fsub.d f29, f29, f29 \n"
        "fsub.d f30, f30, f30 \n"
        "frep.o %[cnt], 8, 0, 0 \n"
        "fmadd.d f3, f0, f1, f3 \n"
        "fmadd.d f4, f0, f1, f4 \n"
        "fmadd.d f5, f0, f1, f5 \n"
        "fmadd.d f6, f0, f1, f6 \n"
        "fmadd.d f7, f0, f1, f7 \n"
        "fmadd.d f28, f0, f1, f28 \n"
        "fmadd.d f29, f0, f1, f29 \n"
        "fmadd.d f30, f0, f1, f30 \n"
        "fmadd.d f2, f0, f1, f3 \n"
        "fmadd.d f2, f0, f1, f4 \n"
        "fmadd.d f2, f0, f1, f5 \n"
        "fmadd.d f2, f0, f1, f6 \n"
        "fmadd.d f2, f0, f1, f7 \n"
        "fmadd.d f2, f0, f1, f28 \n"
        "fmadd.d f2, f0, f1, f29 \n"
        "fmadd.d f2, f0, f1, f30 \n"
        :
        : [ cnt ] "r"(6), [ cnts ] "r"(7)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f28", "f29", "f30",
          "memory");
    snrt_fpu_fence();
    snrt_ssr_disable();
}

void opt_quant(double *in, double *out, const double *recip) {
    register double magic asm("ft10") = 6755399441055744.0;

    snrt_ssr_loop_1d(SNRT_SSR_DM0, 64, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void *)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 64, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void *)recip);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void *)out);

    snrt_ssr_enable();
    asm volatile(
        "frep.o %[cnt], 6, 0, 0    \n"
        "fmul.d  ft3, ft0, ft1     \n"
        "fmul.d  ft4, ft0, ft1     \n"
        "fadd.d  ft5, ft3, ft10, rne \n"
        "fadd.d  ft6, ft4, ft10, rne \n"
        "fsub.d  ft2, ft5, ft10, rne \n"
        "fsub.d  ft2, ft6, ft10, rne \n"
        :
        : [ cnt ] "r"(31), "f"(magic)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6",
          "memory"  // ft10 holds the magic constant (input)
    );
    snrt_fpu_fence();
    snrt_ssr_disable();
}
