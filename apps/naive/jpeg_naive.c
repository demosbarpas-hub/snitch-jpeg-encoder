// Copyright 2026 Dimosthenis Karamparpas
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "image_data.h"

/* --- Configuration & Macros --- */
#define BANK_SIZE 8
#define NUM_BANKS 32
#define ROW_SIZE (NUM_BANKS * BANK_SIZE) 

#define ALIGN_UP_TCDM(addr) \
    ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)
    
// Fixed output address in L3 DRAM
#define DRAM_OUT_ADDR  0x80208000

/* --- Fixed dimensions of a standard 4:2:0 MCU --- */
#define MCU_W 16                 
#define MCU_H 16                 
#define MCU_TOTAL_PIXELS  (MCU_W * MCU_H)          // Always 256
#define MCU_CHROMA_PIXELS (MCU_TOTAL_PIXELS / 4)   // Always 64

#define MAX(a,b)  (((a)>(b))?(a):(b))
#define MIN(a,b)  (((a)<(b))?(a):(b))
#define CLIP(n, min, max) MIN((MAX((n),(min))), (max))

/* --- Constant DCT & quantization tables (L2) --- */
const double dct_m_l2[8][8] __attribute__((aligned(64))) = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545, -0.097545, -0.277785, -0.415735, -0.490393},
    {0.461940, 0.191342, -0.191342, -0.461940, -0.461940, -0.191342, 0.191342, 0.461940},
    {0.415735, -0.097545, -0.490393, -0.277785, 0.277785, 0.490393, 0.097545, -0.415735},
    {0.353553, -0.353553, -0.353553, 0.353553, 0.353553, -0.353553, -0.353553, 0.353553},
    {0.277785, -0.490393, 0.097545, 0.415735, -0.415735, -0.097545, 0.490393, -0.277785},
    {0.191342, -0.461940, 0.461940, -0.191342, -0.191342, 0.461940, -0.461940, 0.191342},
    {0.097545, -0.277785, 0.415735, -0.490393, 0.490393, -0.415735, 0.277785, -0.097545}
};

const double quant_recip_lum_l2[64] __attribute__((aligned(64))) = {
    0.062500, 0.090909, 0.100000, 0.062500, 0.041667, 0.025000, 0.019608, 0.016393,
    0.083333, 0.083333, 0.071429, 0.052632, 0.038462, 0.018182, 0.017544, 0.014493,
    0.071429, 0.076923, 0.062500, 0.038462, 0.025000, 0.017544, 0.014493, 0.012048,
    0.071429, 0.058824, 0.045455, 0.029412, 0.018182, 0.010417, 0.011364, 0.013158,
    0.055556, 0.045455, 0.027027, 0.017857, 0.014706, 0.009009, 0.009709, 0.011236,
    0.041667, 0.028571, 0.018182, 0.015625, 0.011364, 0.009434, 0.008850, 0.010101,
    0.020408, 0.015625, 0.012500, 0.011111, 0.009174, 0.007812, 0.007752, 0.009259,
    0.013889, 0.010870, 0.010526, 0.010309, 0.009009, 0.007937, 0.008065, 0.009434
};

const double quant_recip_chr_l2[64] __attribute__((aligned(64))) = {
    0.058824, 0.055556, 0.041667, 0.021277, 0.010101, 0.010101, 0.010101, 0.010101,
    0.055556, 0.047619, 0.038462, 0.015152, 0.010101, 0.010101, 0.010101, 0.010101,
    0.041667, 0.038462, 0.017857, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.021277, 0.015152, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101
};

/* Quantization tables (integers, for the DQT headers) */
int luma_quantizer[64] = {
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99
};

int chroma_quantizer[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

/* Standard JPEG Zig-Zag Map */
const uint8_t zigzag_map[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// ========================================================================
// STANDARD HUFFMAN TABLES
// ========================================================================
const uint16_t std_luma_dc_code[12] = { 0, 2, 3, 4, 5, 6, 14, 30, 62, 126, 254, 510 };
const uint8_t  std_luma_dc_len[12]  = { 2, 3, 3, 3, 3, 3,  4,  5,  6,   7,   8,   9 };
const uint8_t  std_luma_dc_bits[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
const uint8_t  std_luma_dc_vals[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

const uint16_t std_chroma_dc_code[12] = { 0, 1, 2, 6, 14, 30, 62, 126, 254, 510, 1022, 2046 };
const uint8_t  std_chroma_dc_len[12]  = { 2, 2, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11 };
const uint8_t  std_chroma_dc_bits[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const uint8_t  std_chroma_dc_vals[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

uint16_t std_luma_ac_code[256];
uint8_t  std_luma_ac_len[256];
uint16_t std_chroma_ac_code[256];
uint8_t  std_chroma_ac_len[256];

const uint8_t luma_ac_bits[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125 };
const uint8_t luma_ac_vals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

const uint8_t chroma_ac_bits[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119 };
const uint8_t chroma_ac_vals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

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
    code = 0; k = 0;
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
    uint8_t  *jpeg_output;      
    uint32_t  byte_pos;         
    uint8_t   bit_buffer;       
    int32_t   num_bits;         
} BitWriter;

void init_bit_writer(BitWriter *writer, uint8_t *target_payload) {
    writer->jpeg_output  = target_payload;
    writer->byte_pos     = 0;
    writer->bit_buffer   = 0;
    writer->num_bits     = 0;
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
        uint8_t padded_byte = (writer->bit_buffer << missing_bits) | ((1 << missing_bits) - 1);
        writer->jpeg_output[writer->byte_pos++] = padded_byte;
        if (padded_byte == 0xFF) {
            writer->jpeg_output[writer->byte_pos++] = 0x00;
        }
        writer->bit_buffer = 0;
        writer->num_bits = 0;
    }
}

void get_huffman_category(int val, int *category, int *code) {
    if (val == 0) { *category = 0; *code = 0; return; }
    int abs_val = (val < 0) ? -val : val;
    int cat = 0;
    while (abs_val > 0) { cat++; abs_val >>= 1; }
    *category = cat;
    if (val < 0) *code = val + (1 << cat) - 1;
    else *code = val;
}

void write_dht_marker(BitWriter *w, const uint8_t bits[], const uint8_t vals[], int num_vals, int tc_th) {
    int length = 2 + 1 + 16 + num_vals; 
    write_byte_out(w, 0xFF); write_byte_out(w, 0xC4); 
    write_byte_out(w, (length >> 8) & 0xFF); write_byte_out(w, length & 0xFF); 
    write_byte_out(w, tc_th); 
    for (int i = 0; i < 16; i++) write_byte_out(w, bits[i]);
    for (int i = 0; i < num_vals; i++) write_byte_out(w, vals[i]);
}

void transform_to_16x16_mcu_tiles(const uint8_t *input_raster, uint8_t *output_mcu_tiled, int img_w, int img_h) {
    int mcus_wide = img_w / 16;
    int mcus_high = img_h / 16;
    int write_idx = 0;

    for (int mcu_row = 0; mcu_row < mcus_high; mcu_row++) {
        for (int mcu_col = 0; mcu_col < mcus_wide; mcu_col++) {
            for (int pixel_y = 0; pixel_y < 16; pixel_y++) {
                for (int pixel_x = 0; pixel_x < 16; pixel_x++) {
                    int global_y = mcu_row * 16 + pixel_y;
                    int global_x = mcu_col * 16 + pixel_x;
                    int global_raster_idx = (global_y * img_w + global_x) * 3;

                    output_mcu_tiled[write_idx + 0] = input_raster[global_raster_idx + 0];
                    output_mcu_tiled[write_idx + 1] = input_raster[global_raster_idx + 1];
                    output_mcu_tiled[write_idx + 2] = input_raster[global_raster_idx + 2];
                    write_idx += 3;
                }
            }
        }
    }
}

// ========================================================================
// NAIVE PIPELINE STAGES
// ========================================================================
void naive_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr, double *lut, int w, int h) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int linear_pixel_idx = row * w + col;
            double R = lut[input_idx[linear_pixel_idx * 3 + 0]];
            double G = lut[input_idx[linear_pixel_idx * 3 + 1]];
            double B = lut[input_idx[linear_pixel_idx * 3 + 2]];

            double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
            double Cb = 0.564 * (B - Y)+128.0;
            double Cr = 0.713 * (R - Y)+128.0;

            int block_row = row / 8;       
            int block_col = col / 8;       
            int pixel_y_inside = row % 8;  
            int pixel_x_inside = col % 8;  
            int blocks_wide = w / 8;
            int tiled_Y_idx = (block_row * blocks_wide + block_col) * 64 + (pixel_y_inside * 8 + pixel_x_inside);

            out_Y[tiled_Y_idx] = Y; 
            out_CbCr[linear_pixel_idx * 2 + 0] = Cb; 
            out_CbCr[linear_pixel_idx * 2 + 1] = Cr; 
        }
    }
}

void naive_chroma_420(double *in_cbcr, double *out_cb_f, double *out_cr_f, int w, int h) {
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; x += 2) {
            double sum_cb = 0.0, sum_cr = 0.0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int in_idx = ((y + dy) * w + (x + dx)) * 2;
                    sum_cb += in_cbcr[in_idx + 0]; 
                    sum_cr += in_cbcr[in_idx + 1]; 
                }
            }
            int block_row = (y / 2) / 8, block_col = (x / 2) / 8;
            int pixel_y = (y / 2) % 8, pixel_x = (x / 2) % 8;
            int tiled_idx = (block_row * ((w / 2) / 8) + block_col) * 64 + (pixel_y * 8 + pixel_x);
            out_cb_f[tiled_idx] = sum_cb * 0.25; 
            out_cr_f[tiled_idx] = sum_cr * 0.25; 
        }
    }
}

void naive_dct(double *in, double *out, const double dct_m[8][8]) {
    double s[8][8], tmp[8][8];
    for (int i = 0; i < 64; i++) s[i/8][i%8] = in[i]-128.0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += s[r][k] * dct_m[c][k];
            tmp[r][c] = sum;
        }
    }
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += dct_m[r][k] * tmp[k][c];
            out[r * 8 + c] = sum;
        }
    }
}

void naive_quant(double *in, int16_t *out, const double *recip) {
    for (int i = 0; i < 64; i++) {
        out[i] = (int16_t)nearbyint(in[i] * recip[i]);
    }
}

void naive_zigzag_and_diff(int16_t *in_blocks, int16_t *out_blocks, int *prev_dc, int num_blocks) {
    for (int b = 0; b < num_blocks; b++) {
        int16_t shuffled[64];
        for (int i = 0; i < 64; i++) shuffled[i] = in_blocks[b * 64 + zigzag_map[i]];
        int current_dc = shuffled[0];
        shuffled[0] = current_dc - *prev_dc;
        *prev_dc = current_dc;
        for (int i = 0; i < 64; i++) out_blocks[b * 64 + i] = shuffled[i];
    }
}

void naive_huffman_rle_encode_block(int16_t *zigzag_block, BitWriter *writer, int is_chroma) {
    int category, v_code;
    
    get_huffman_category(zigzag_block[0], &category, &v_code);
    if (!is_chroma) {
        write_bits(writer, std_luma_dc_code[category], std_luma_dc_len[category]);
    } else {
        write_bits(writer, std_chroma_dc_code[category], std_chroma_dc_len[category]);
    }
    write_bits(writer, v_code, category);

    int r = 0; 
    for (int i = 1; i < 64; i++) {
        if (zigzag_block[i] == 0) {
            r++;
        } else {
            while (r > 15) {
                if (!is_chroma) write_bits(writer, std_luma_ac_code[0xF0], std_luma_ac_len[0xF0]);
                else write_bits(writer, std_chroma_ac_code[0xF0], std_chroma_ac_len[0xF0]);
                r -= 16;
            }
            get_huffman_category(zigzag_block[i], &category, &v_code);
            uint8_t rs_sym = (r << 4) | category;
            
            if (!is_chroma) {
                write_bits(writer, std_luma_ac_code[rs_sym], std_luma_ac_len[rs_sym]);
            } else {
                write_bits(writer, std_chroma_ac_code[rs_sym], std_chroma_ac_len[rs_sym]);
            }
            write_bits(writer, v_code, category);
            r = 0;
        }
    }
    if (r > 0) {
        if (!is_chroma) write_bits(writer, std_luma_ac_code[0x00], std_luma_ac_len[0x00]);
        else write_bits(writer, std_chroma_ac_code[0x00], std_chroma_ac_len[0x00]);
    }
}

int main() {
    
    uint16_t width  = IMG_WIDTH;  // 48
    uint16_t height = IMG_HEIGHT; // 48
    
    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = (uintptr_t)snrt_l1_next();
    
    volatile uint32_t *shared_size = (volatile uint32_t *)base;
    
    double *lut          = (double *)ALIGN_UP_TCDM((uintptr_t)shared_size + sizeof(uint32_t));
    uint8_t *input_idx   = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));
    
    double *naive_Y      = (double *)ALIGN_UP_TCDM((uintptr_t)input_idx + MCU_TOTAL_PIXELS * 3);
    double *naive_CbCr   = (double *)ALIGN_UP_TCDM((uintptr_t)naive_Y + MCU_TOTAL_PIXELS * sizeof(double));
    double *naive_Cb_420 = (double *)ALIGN_UP_TCDM((uintptr_t)naive_CbCr + MCU_TOTAL_PIXELS * 2 * sizeof(double));
    double *naive_Cr_420 = (double *)ALIGN_UP_TCDM((uintptr_t)naive_Cb_420 + MCU_CHROMA_PIXELS * sizeof(double));
    
    double *naive_dct_Y  = (double *)ALIGN_UP_TCDM((uintptr_t)naive_Cr_420 + MCU_CHROMA_PIXELS * sizeof(double));
    double *naive_dct_Cb = (double *)ALIGN_UP_TCDM((uintptr_t)naive_dct_Y + MCU_TOTAL_PIXELS * sizeof(double));
    double *naive_dct_Cr = (double *)ALIGN_UP_TCDM((uintptr_t)naive_dct_Cb + MCU_CHROMA_PIXELS * sizeof(double));

    int16_t *naive_quant_Y  = (int16_t *)ALIGN_UP_TCDM((uintptr_t)naive_dct_Cr + MCU_CHROMA_PIXELS * sizeof(double));
    int16_t *naive_quant_Cb = (int16_t *)ALIGN_UP_TCDM((uintptr_t)naive_quant_Y + MCU_TOTAL_PIXELS * sizeof(int16_t));
    int16_t *naive_quant_Cr = (int16_t *)ALIGN_UP_TCDM((uintptr_t)naive_quant_Cb + MCU_CHROMA_PIXELS * sizeof(int16_t));

    int16_t *zz_diff_Y      = (int16_t *)ALIGN_UP_TCDM((uintptr_t)naive_quant_Cr + MCU_CHROMA_PIXELS * sizeof(int16_t));
    int16_t *zz_diff_Cb     = (int16_t *)ALIGN_UP_TCDM((uintptr_t)zz_diff_Y + MCU_TOTAL_PIXELS * sizeof(int16_t));
    int16_t *zz_diff_Cr     = (int16_t *)ALIGN_UP_TCDM((uintptr_t)zz_diff_Cb + MCU_CHROMA_PIXELS * sizeof(int16_t));

    uint8_t *jpeg_output_payload = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)zz_diff_Cr + MCU_CHROMA_PIXELS * sizeof(int16_t));

    double *l1_matrix    = (double *)ALIGN_UP_TCDM((uintptr_t)jpeg_output_payload + 4096); 
    double *l1_recip_lum = (double *)ALIGN_UP_TCDM((uintptr_t)l1_matrix + 64 * sizeof(double));
    double *l1_recip_chr = (double *)ALIGN_UP_TCDM((uintptr_t)l1_recip_lum + 64 * sizeof(double));

    // L3 streaming tracking variables (address & global counter)
    uint8_t *l3_write_ptr = (uint8_t *)DRAM_OUT_ADDR;
    uint32_t total_compressed_bytes = 0;

    // ========================================================================
    // ONLY THE DM CORE LOADS THE TABLES AND THE IMAGE INTO L1 VIA DMA
    // ========================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1_matrix, (void*)dct_m_l2, 64 * sizeof(double));
        snrt_dma_start_1d(l1_recip_lum, (void*)quant_recip_lum_l2, 64 * sizeof(double));
        snrt_dma_start_1d(l1_recip_chr, (void*)quant_recip_chr_l2, 64 * sizeof(double));
        
        snrt_dma_wait_all();
    }
    
    snrt_cluster_hw_barrier();

    // ========================================================================
    // COMPUTE CORE 0 PERFORMS INITIALIZATION AND WRITES THE HEADERS
    // ========================================================================
    BitWriter writer;
    int mcus_wide = width / 16;
    int mcus_high = height / 16;
    int total_mcus = mcus_wide * mcus_high; 
    int prev_dc_Y = 0, prev_dc_Cb = 0, prev_dc_Cr = 0;

    // PROFILING VARIABLES
    uint32_t cycles_ycbcr   = 0;
    uint32_t cycles_chroma  = 0;
    uint32_t cycles_dct     = 0;
    uint32_t cycles_quant   = 0;
    uint32_t cycles_zigzag  = 0;
    uint32_t cycles_huffman = 0;
    uint32_t t_start, t_end;

    if (snrt_is_compute_core() && core_id == 0) {
        init_standard_ac_tables();
        
        for (int i = 0; i < 256; i++) lut[i] = (double)i;

        init_bit_writer(&writer, jpeg_output_payload);

        // --- SOI & APP0 Markers ---
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xD8); 
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xE0); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x10); 
        write_byte_out(&writer, 0x4A); write_byte_out(&writer, 0x46); write_byte_out(&writer, 0x49); write_byte_out(&writer, 0x46); write_byte_out(&writer, 0x00); 
        write_byte_out(&writer, 0x01); write_byte_out(&writer, 0x01); write_byte_out(&writer, 0x00); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x48); write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x48); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x00); 

        // DQT Luma & Chroma
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xDB); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x43); 
        write_byte_out(&writer, 0x00); 
        for (int i = 0; i < 64; i++) {
            write_byte_out(&writer, (uint8_t)luma_quantizer[zigzag_map[i]]);
        }
        
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xDB); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x43); 
        write_byte_out(&writer, 0x01); 
        for (int i = 0; i < 64; i++) {
            write_byte_out(&writer, (uint8_t)chroma_quantizer[zigzag_map[i]]);
        }

        // DHT Markers
        write_dht_marker(&writer, std_luma_dc_bits, std_luma_dc_vals, 0x0C, 0x00);   
        write_dht_marker(&writer, luma_ac_bits, luma_ac_vals, 0xA2, 0x10);          
        write_dht_marker(&writer, std_chroma_dc_bits, std_chroma_dc_vals, 0x0C, 0x01); 
        write_dht_marker(&writer, chroma_ac_bits, chroma_ac_vals, 0xA2, 0x11);      

        // SOF0 Marker
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xC0); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x11); 
        write_byte_out(&writer, 0x08);                                 
        write_byte_out(&writer, (height >> 8) & 0xFF); write_byte_out(&writer, height & 0xFF); 
        write_byte_out(&writer, (width >> 8) & 0xFF); write_byte_out(&writer, width & 0xFF);   
        write_byte_out(&writer, 0x03);                                 
        write_byte_out(&writer, 0x01); write_byte_out(&writer, 0x22); write_byte_out(&writer, 0x00); 
        write_byte_out(&writer, 0x02); write_byte_out(&writer, 0x11); write_byte_out(&writer, 0x01); 
        write_byte_out(&writer, 0x03); write_byte_out(&writer, 0x11); write_byte_out(&writer, 0x01); 

        // SOS Marker
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xDA); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x0C); 
        write_byte_out(&writer, 0x03);                                 
        write_byte_out(&writer, 0x01); write_byte_out(&writer, 0x00); 
        write_byte_out(&writer, 0x02); write_byte_out(&writer, 0x11); 
        write_byte_out(&writer, 0x03); write_byte_out(&writer, 0x11); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x3F); write_byte_out(&writer, 0x00); 

        // Store the size of the initial headers in TCDM
        *shared_size = writer.byte_pos;
    }

    snrt_cluster_hw_barrier();

    // ========================================================================
    // HEADER TRANSFER: the DM core flushes the headers to DRAM before the loop
    // ========================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l3_write_ptr, jpeg_output_payload, *shared_size);
        snrt_dma_wait_all();
        l3_write_ptr += *shared_size; // Advance the L3 pointer
    }
    snrt_cluster_hw_barrier();

    // Compute core 0 updates the running total and resets the L1 buffer
    if (snrt_is_compute_core() && core_id == 0) {
        total_compressed_bytes += *shared_size;
        writer.byte_pos = 0; // L1 buffer is clean again (index 0) for the loop
    }
    snrt_cluster_hw_barrier();

    // ========================================================================
    // MAIN LOOP (STREAMING ONE MCU AT A TIME)
    // ========================================================================
    for (uint32_t mcu = 0; mcu < total_mcus; mcu++) {
        
        // ----------------------------------------------------------------
        // [DMA PART] - The DM core fetches one tile at a time into L1
        // ----------------------------------------------------------------
        if (snrt_is_dm_core()) {
			int mcu_row = mcu / mcus_wide;
			int mcu_col = mcu % mcus_wide;
			
			uint8_t *dst = input_idx; 
			void *src = (void*)&lena_pixels[(mcu_row * 16 * width + mcu_col * 16) * 3];
			
			snrt_dma_start_2d(dst, src, 16 * 3, 16 * 3, width * 3, 16, 0);
			snrt_dma_wait_all();
		}

        // BARRIER 1: Wait for the DMA to fill the input_idx buffer
        snrt_cluster_hw_barrier();

        // ----------------------------------------------------------------
        // [CORE PART] - The compute core processes the tile
        // ----------------------------------------------------------------
        if (snrt_is_compute_core() && core_id == 0) {
            
            printf("Processing MCU %d/%d...\n", mcu + 1, total_mcus);

            // --- STAGE 1: YCbCr Conversion ---
            t_start = snrt_mcycle();
            naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, 16, 16);
            t_end = snrt_mcycle();
            cycles_ycbcr += (t_end - t_start);

            // --- STAGE 2: Chroma Subsampling ---
            t_start = snrt_mcycle();
            naive_chroma_420(naive_CbCr, naive_Cb_420, naive_Cr_420, 16, 16);
            t_end = snrt_mcycle();
            cycles_chroma += (t_end - t_start);
            
            // --- STAGE 3: 2D DCT ---
            t_start = snrt_mcycle();
            for (int b = 0; b < 4; b++) {
                naive_dct(&naive_Y[b * 64], &naive_dct_Y[b * 64], (const double(*)[8])l1_matrix);
            }
            naive_dct(naive_Cb_420, naive_dct_Cb, (const double(*)[8])l1_matrix);
            naive_dct(naive_Cr_420, naive_dct_Cr, (const double(*)[8])l1_matrix);
            t_end = snrt_mcycle();
            cycles_dct += (t_end - t_start);
            
            // --- STAGE 4: Quantization ---
            t_start = snrt_mcycle();
            for (int b = 0; b < 4; b++) {
                naive_quant(&naive_dct_Y[b * 64], &naive_quant_Y[b * 64], l1_recip_lum);
            }
            naive_quant(naive_dct_Cb, naive_quant_Cb, l1_recip_chr);
            naive_quant(naive_dct_Cr, naive_quant_Cr, l1_recip_chr);
            t_end = snrt_mcycle();
            cycles_quant += (t_end - t_start);

            // --- STAGE 5: Zig-Zag Parsing & DC Diff ---
            t_start = snrt_mcycle();
            naive_zigzag_and_diff(naive_quant_Y, zz_diff_Y, &prev_dc_Y, 4);
            naive_zigzag_and_diff(naive_quant_Cb, zz_diff_Cb, &prev_dc_Cb, 1);
            naive_zigzag_and_diff(naive_quant_Cr, zz_diff_Cr, &prev_dc_Cr, 1);
            t_end = snrt_mcycle();
            cycles_zigzag += (t_end - t_start);

            // --- STAGE 6: Huffman Entropy Coding (always writes from index 0 of the L1 buffer) ---
            t_start = snrt_mcycle();
            for (int b = 0; b < 4; b++) {
                naive_huffman_rle_encode_block(&zz_diff_Y[b * 64], &writer, 0); 
            }
            naive_huffman_rle_encode_block(zz_diff_Cb, &writer, 1); 
            naive_huffman_rle_encode_block(zz_diff_Cr, &writer, 1);
            t_end = snrt_mcycle();
            cycles_huffman += (t_end - t_start);

            // Store the number of bytes produced by THIS MCU in TCDM
            *shared_size = writer.byte_pos;
        }

        // BARRIER 2: Wait for compute core 0 to finish the current MCU
        snrt_cluster_hw_barrier();

        // ----------------------------------------------------------------
        // [MCU DMA OUT] - The DM core sends ONLY the new bytes to DRAM
        // ----------------------------------------------------------------
        if (snrt_is_dm_core()) {
            if (*shared_size > 0) {
                snrt_dma_start_1d(l3_write_ptr, jpeg_output_payload, *shared_size);
                snrt_dma_wait_all();
                l3_write_ptr += *shared_size; // Advance the DRAM pointer
            }
        }
        snrt_cluster_hw_barrier();

        // CLEANUP: compute core 0 records statistics and resets the L1 buffer index
        if (snrt_is_compute_core() && core_id == 0) {
            total_compressed_bytes += *shared_size;
            writer.byte_pos = 0; // Overwrite from index 0 in the next iteration
        }
        snrt_cluster_hw_barrier();
    }

    // ========================================================================
    // CLOSING PHASE (POST-LOOP)
    // ========================================================================
    if (snrt_is_compute_core() && core_id == 0) {
        flush_bit_writer(&writer); // Flush the remaining bits (fractional byte)
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xD9); // Write EOI marker
        
        *shared_size = writer.byte_pos; // Size of the tail
    }

    // BARRIER: The DM core waits for the tail to be complete in L1
    snrt_cluster_hw_barrier();

    // ========================================================================
    // FINAL DMA OUT: transfer the EOI and the flush byte to DRAM
    // ========================================================================
    if (snrt_is_dm_core()) {
        if (*shared_size > 0) {
            snrt_dma_start_1d(l3_write_ptr, jpeg_output_payload, *shared_size);
            snrt_dma_wait_all();
        }
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && core_id == 0) {
        total_compressed_bytes += *shared_size; // Add the final bytes as well
    }
    snrt_cluster_hw_barrier();

    // ========================================================================
    // POST-LOOP PHASE: PRINT RESULTS FROM L3 DRAM
    // ========================================================================
    if (snrt_is_compute_core() && core_id == 0) {
        uint8_t *l3_output = (uint8_t *)DRAM_OUT_ADDR;
        
        // PRINT CYCLE STATISTICS
        printf("\n================  PERFORMANCE PROFILE  ===============\n");
        printf("  Total MCUs Processed      : %d\n", total_mcus);
        printf("  Stage 1: YCbCr Conversion : %u cycles\n", cycles_ycbcr);
        printf("  Stage 2: Subsampling 4:2:0: %u cycles\n", cycles_chroma);
        printf("  Stage 3: Forward 2D-DCT   : %u cycles\n", cycles_dct);
        printf("  Stage 4: Quantization     : %u cycles\n", cycles_quant);
        printf("  Stage 5: ZigZag & Diff    : %u cycles\n", cycles_zigzag);
        printf("  Stage 6: Huffman Encoding : %u cycles\n", cycles_huffman);
        printf("------------------------------------------------------\n");
        printf("  Total Compute Pipeline    : %u cycles\n", 
               cycles_ycbcr + cycles_chroma + cycles_dct + cycles_quant + cycles_zigzag + cycles_huffman);
        printf("=====================================================\n\n");
        
        // Print the real total size written to DRAM
        printf("  Final Valid JPG File Size  : %u bytes\n", total_compressed_bytes);
        printf("=====================================================\n");
        
        // Read back and hex-dump directly from L3 DRAM
        printf("\n--- START JPEG HEX DUMP (Read directly from L3 DRAM) ---\n");
        for (uint32_t i = 0; i < total_compressed_bytes; i++) {
            printf("%02X", l3_output[i]);
            if ((i + 1) % 32 == 0) printf("\n"); 
        }
        printf("\n--- END JPEG HEX DUMP ---\n");
    }

    snrt_cluster_hw_barrier();
    if (core_id != 0) while (1) asm volatile("wfi");
    return 0;
}
