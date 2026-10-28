// Copyright 2026 Dimosthenis Karamparpas
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "image_data.h"

/* --- Compression configuration & macros --- */
#define JPEG_QUALITY 50      // Compression quality (1-100)

/* 1 = Hybrid: FP16 stages 1-4, then widen FP16->FP64 (flh) + FP64 ISSR/copift zig-zag
   0 = Original FP16 scalar unpack (for A/B comparison of cycles and hex dump) */
#define USE_HYBRID_ZIGZAG 1

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

#define BLOCK_W  8
#define BLOCK_H  8
#define BLOCKS_X (MCU_W / BLOCK_W)  
#define BLOCKS_Y (MCU_H / BLOCK_H)  

static inline int tiled_Y_idx(int row, int col) {
    int block_row = row / BLOCK_H;
    int block_col = col / BLOCK_W;
    int local_row = row % BLOCK_H;
    int local_col = col % BLOCK_W;
    return (block_row * BLOCKS_X + block_col) * (BLOCK_W * BLOCK_H)
           + local_row * BLOCK_W + local_col;
}

typedef uint16_t float16_t;

static inline float16_t float_to_fp16(float x) {
    union { float f; uint32_t u; } v;
    asm volatile("fcvt.h.s %0, %1" : "=f"(v.f) : "f"(x));
    return (float16_t)(v.u & 0xFFFF);
}

static inline float fp16_to_float(float16_t x) {
    union { float f; uint32_t u; } v;
    v.u = 0xFFFF0000 | (uint32_t)x;
    float result;
    asm volatile("fcvt.s.h %0, %1" : "=f"(result) : "f"(v.f));
    return result;
}

/* =========================================================================
   CONSTANT DCT, QUANTIZATION & HUFFMAN TABLES
========================================================================= */
const uint64_t dct_m[16] __attribute__((aligned(64))) = {
    0x35a835a835a835a8, 0x35a835a835a835a8, // Row 0
    0x2e3e347236a737d9, 0xb7d9b6a7b472ae3e, // Row 1
    0xb764b21f321f3764, 0x3764321fb21fb764, // Row 2
    0xb472b7d9ae3e36a7, 0xb6a72e3e37d93472, // Row 3
    0x35a8b5a8b5a835a8, 0x35a8b5a8b5a835a8, // Row 4
    0x36a72e3eb7d93472, 0xb47237d9ae3eb6a7, // Row 5
    0xb21f3764b764321f, 0x321fb7643764b21f, // Row 6
    0xb7d936a7b4722e3e, 0xae3e3472b6a737d9  // Row 7
};

const uint64_t dct_m_bcast[64] __attribute__((aligned(64))) = {
    // --- Row 0 ---
    0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8,
    0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8,
    // --- Row 1 ---
    0x37d937d937d937d9, 0x36a736a736a736a7, 0x3472347234723472, 0x2e3e2e3e2e3e2e3e,
    0xae3eae3eae3eae3e, 0xb472b472b472b472, 0xb6a7b6a7b6a7b6a7, 0xb7d9b7d9b7d9b7d9,
    // --- Row 2 ---
    0x3764376437643764, 0x321f321f321f321f, 0xb21fb21fb21fb21f, 0xb764b764b764b764,
    0xb764b764b764b764, 0xb21fb21fb21fb21f, 0x321f321f321f321f, 0x3764376437643764,
    // --- Row 3 ---
    0x36a736a736a736a7, 0xae3eae3eae3eae3e, 0xb7d9b7d9b7d9b7d9, 0xb472b472b472b472,
    0x3472347234723472, 0x37d937d937d937d9, 0x2e3e2e3e2e3e2e3e, 0xb6a7b6a7b6a7b6a7,
    // --- Row 4 ---
    0x35a835a835a835a8, 0xb5a8b5a8b5a8b5a8, 0xb5a8b5a8b5a8b5a8, 0x35a835a835a835a8,
    0x35a835a835a835a8, 0xb5a8b5a8b5a8b5a8, 0xb5a8b5a8b5a8b5a8, 0x35a835a835a835a8,
    // --- Row 5 ---
    0x3472347234723472, 0xb7d9b7d9b7d9b7d9, 0x2e3e2e3e2e3e2e3e, 0x36a736a736a736a7,
    0xb6a7b6a7b6a7b6a7, 0xae3eae3eae3eae3e, 0x37d937d937d937d9, 0xb472b472b472b472,
    // --- Row 6 ---
    0x321f321f321f321f, 0xb764b764b764b764, 0x3764376437643764, 0xb21fb21fb21fb21f,
    0xb21fb21fb21fb21f, 0x3764376437643764, 0xb764b764b764b764, 0x321f321f321f321f,
    // --- Row 7 ---
    0x2e3e2e3e2e3e2e3e, 0xb472b472b472b472, 0x36a736a736a736a7, 0xb7d9b7d9b7d9b7d9,
    0x37d937d937d937d9, 0xb6a7b6a7b6a7b6a7, 0x3472347234723472, 0xae3eae3eae3eae3e
};

const uint8_t std_luma_q[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,
    24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

const uint8_t std_chroma_q[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

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

/* --- Bit-writer --- */
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

void build_quant_tables(const uint8_t *base_table, float16_t *recip_fp16, double *recip_f64, int quality) {
    if (quality <= 0) quality = 1;
    if (quality > 100) quality = 100;
    int scale = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);
    
    for (int i = 0; i < 64; i++) {
        int temp = ((int)base_table[i] * scale + 50) / 100;
        if (temp <= 0) temp = 1;
        if (temp > 255) temp = 255;
        
        recip_f64[i] = 1.0 / (double)temp;
        recip_fp16[i] = float_to_fp16((float)recip_f64[i]);
    }
}

/* =========================================================================
   OPTIMIZED ASSEMBLY PIPELINE KERNELS (YCbCr, Subsampling, DCT, Quant)
========================================================================= */
void opt_YCBCR_conversion(uint8_t *input_idx, float16_t *out_Y, float16_t *out_CbCr, double *lut, int total_pixels) {
    float c_r_f  = 0.299f;  float c_g_f  = 0.587f;  float c_b_f  = 0.114f;
    float c_cb_f = 0.141f;  float c_cr_f = 0.17825f;  
    asm volatile(
        "fcvt.h.s f3, %0 \n" "fcvt.h.s f4, %1 \n" "fcvt.h.s f5, %2 \n" "fcvt.h.s f6, %3 \n" "fcvt.h.s f7, %4 \n"
        :: "f"(c_r_f), "f"(c_g_f), "f"(c_b_f), "f"(c_cb_f), "f"(c_cr_f) : "f3","f4","f5","f6","f7"
    );
    snrt_issr_read(SNRT_SSR_DM0, lut, input_idx, total_pixels * 3, SNRT_SSR_IDXSIZE_U8);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 2, 2, 8, 2, 8, 8*2*8, 8*2, 8*8*2*2);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_4D, out_Y);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, total_pixels / 2, sizeof(uint64_t));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_CbCr);
    snrt_ssr_enable();
    asm volatile(
        "frep.o %[n], 20, 0, 0 \n"
        "fmv.d f8, f0 \n fmv.d f9, f0 \n fmv.d f10, f0 \n"
        "vfcpka.h.d f8, f8, f0 \n vfcpka.h.d f9, f9, f0 \n vfcpka.h.d f10, f10, f0 \n"
        "fmv.d f14, f0 \n fmv.d f15, f0 \n fmv.d f16, f0 \n"
        "vfcpkb.h.d f8, f14, f0 \n vfcpkb.h.d f9, f15, f0 \n vfcpkb.h.d f10, f16, f0 \n"
        "vfmul.r.h f11, f8, f3 \n vfmac.r.h f11, f9, f4 \n vfmac.r.h f11, f10, f5 \n"
        "vfsub.h f10, f10, f11 \n vfsub.h f8, f8, f11 \n"
        "fmv.d f1, f11 \n" "vfmul.r.h f2, f10, f6 \n vfmul.r.h f2, f8, f7 \n"
        :: [n] "r"(total_pixels / 4 - 1) : "f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f14","f15","f16","memory"
    );
    snrt_fpu_fence(); snrt_ssr_disable();
}

void opt_chroma_420_2d_ssr(double *in_cbcr, float16_t *out_cb_f, float16_t *out_cr_f) {
    double zero_val = 0.0; 
    double *p_zero = &zero_val;

    // Pack 128.0f in FP16 (0x5800) across all 4 vector lanes
    float16_t val128 = float_to_fp16(128.0f); // 0x5800
    uint64_t c128_vec __attribute__((aligned(8))) = 
        ((uint64_t)val128 << 48) | ((uint64_t)val128 << 32) | 
        ((uint64_t)val128 << 16) | val128;
    uint64_t *p_c128 = &c128_vec;

    uintptr_t base_addr = (uintptr_t)in_cbcr;
    void *p_top = (void*)(base_addr + 0); 
    void *p_bot = (void*)(base_addr + 64);

    snrt_ssr_loop_2d(SNRT_SSR_DM0, 8, 8, 8, 128); snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, p_top);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 8, 8, 8, 128); snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, p_bot);
    snrt_ssr_enable();
    
    uint32_t alu_cb1, alu_cr1, alu_cb2, alu_cr2; 
    uint32_t iters = 15;

    asm volatile(
        "fld      f31, 0(%[zero]) \n"   // Load 0.0f vector into f31
        "fld      f30, 0(%[c128]) \n"   // Load 128.0f vector into f30
        "csrsi    0x801, 0x1 \n"      
        "1: \n"
        "fmv.d    f4, f0 \n"  "fmv.d    f5, f1 \n" "fmv.d    f8, f0 \n"  "fmv.d    f9, f1 \n"  
        "fmv.d    f12, f0 \n" "fmv.d    f13, f1 \n" "fmv.d    f16, f0 \n" "fmv.d    f17, f1 \n" 
        "vfadd.h  f6,  f4,  f5 \n" "vfadd.h  f10, f8,  f9 \n"
        "vfadd.h  f14, f12, f13 \n" "vfadd.h  f18, f16, f17 \n"
        "fmv.d    f7,  f31 \n" "fmv.d    f11, f31 \n" "fmv.d    f15, f31 \n" "fmv.d    f19, f31 \n" // Reset to 0.0f
        "vfsum.h  f7,  f6 \n" "vfsum.h  f11, f10 \n" "vfsum.h  f15, f14 \n" "vfsum.h  f19, f18 \n" // Sum elements onto 0.0f
        "vfadd.h  f7,  f7,  f30 \n" "vfadd.h  f11, f11, f30 \n"                                     // Add 128.0f vector
        "vfadd.h  f15, f15, f30 \n" "vfadd.h  f19, f19, f30 \n"                                     // Add 128.0f vector
        "fmv.x.w  %[alu_cb1], f7 \n" "fmv.x.w  %[alu_cr1], f11 \n" "fmv.x.w  %[alu_cb2], f15 \n" "fmv.x.w  %[alu_cr2], f19 \n"
        "sw       %[alu_cb1], 0(%[out_cb]) \n" "sw       %[alu_cb2], 4(%[out_cb]) \n"
        "sw       %[alu_cr1], 0(%[out_cr]) \n" "sw       %[alu_cr2], 4(%[out_cr]) \n"
        "addi     %[out_cb], %[out_cb], 8 \n" "addi     %[out_cr], %[out_cr], 8 \n"
        "addi     %[iters], %[iters], -1 \n" "bgez     %[iters], 1b \n" 
        "csrci    0x801, 0x1 \n"
        : [out_cb] "+r"(out_cb_f), [out_cr] "+r"(out_cr_f), 
          [alu_cb1] "=&r"(alu_cb1), [alu_cr1] "=&r"(alu_cr1), 
          [alu_cb2] "=&r"(alu_cb2), [alu_cr2] "=&r"(alu_cr2), 
          [iters] "+r"(iters)
        : [zero] "r"(p_zero), [c128] "r"(p_c128)
        : "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f30", "f31", "memory"
    );
    snrt_ssr_disable();
}

void simd_dct_pass1(float16_t *in, float16_t *out, float16_t *tmp, const float16_t *dct_m_param) {
    float16_t off16 = float_to_fp16(-128.0f);
    uint64_t off_vec __attribute__((aligned(8))) = ((uint64_t)off16 << 48) | ((uint64_t)off16 << 32) | ((uint64_t)off16 << 16) | off16;
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 16, 8); snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 16, 8); snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)out);
    snrt_ssr_enable();
    asm volatile("fld f10, 0(%0) \n" "frep.o %[n], 1, 0, 0 \n" "vfadd.h f1, f0, f10 \n" :: "r"(&off_vec), [n]"r"(15) : "f1", "f0", "f10", "memory");
    snrt_fpu_fence(); snrt_ssr_disable();

    snrt_ssr_loop_3d(SNRT_SSR_DM0, 2, 8, 8, 8, 0, 16); snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, (void*)out);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0);       snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, (void*)dct_m_param);
    snrt_ssr_enable();
    double zero_val = 0.0; double *p_zero = &zero_val;
    uint32_t alu_f0, alu_f1, alu_f2, alu_f3; uint32_t alu_idx = 16;
    asm volatile(
        "fld f31, 0(%[zero]) \n"  "csrsi 0x801, 0x1 \n"  
        "1: \n"
        "vfmul.h f4, f0, f1 \n" "vfmul.h f5, f0, f1 \n" "vfmul.h f8, f0, f1 \n" "vfmul.h f9, f0, f1 \n"  
        "vfmul.h f12, f0, f1 \n" "vfmul.h f13, f0, f1 \n" "vfmul.h f16, f0, f1 \n" "vfmul.h f17, f0, f1 \n"  
        "vfadd.h f5, f5, f4 \n" "vfadd.h f9, f9, f8 \n" "vfadd.h f13, f13, f12 \n" "vfadd.h f17, f17, f16 \n"  
        "fmv.d f4, f31 \n" "fmv.d f8, f31 \n" "fmv.d f12, f31 \n" "fmv.d f16, f31 \n"
        "vfsum.h f4, f5 \n" "vfsum.h f8, f9 \n" "vfsum.h f12, f13 \n" "vfsum.h f16, f17 \n"  
        "fmv.d f5, f31 \n" "fmv.d f9, f31 \n" "fmv.d f13, f31 \n" "fmv.d f17, f31 \n"
        "vfsum.h f5, f4 \n" "vfsum.h f9, f8 \n" "vfsum.h f13, f12 \n" "vfsum.h f17, f16 \n"  
        "fmv.x.w %[alu_f0], f5 \n" "fmv.x.w %[alu_f1], f9 \n" "fmv.x.w %[alu_f2], f13 \n" "fmv.x.w %[alu_f3], f17 \n"
        "sh %[alu_f0], 0(%[tmp]) \n" "sh %[alu_f1], 2(%[tmp]) \n" "sh %[alu_f2], 4(%[tmp]) \n" "sh %[alu_f3], 6(%[tmp]) \n"
        "addi %[tmp], %[tmp], 8 \n" "addi %[alu_idx], %[alu_idx], -1 \n" "bnez %[alu_idx], 1b \n" "csrci 0x801, 0x1 \n"  
        : [tmp] "+r"(tmp), [alu_idx] "+r"(alu_idx), [alu_f0] "=&r"(alu_f0), [alu_f1] "=&r"(alu_f1), [alu_f2] "=&r"(alu_f2), [alu_f3] "=&r"(alu_f3)
        : [zero] "r"(p_zero) : "f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f31","memory"
    );
    snrt_ssr_disable();
}

void simd_dct_pass2(float16_t *in, float16_t *out, uint64_t *mat_bcast) {
    snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, 64, 0, 8); snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, mat_bcast);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0); snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, in);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 16, 8);       snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out);
    snrt_ssr_enable();
    uint32_t iters = 8;
    asm volatile(
        "1: \n"
        "vfmul.h f4, f0, f1 \n" "vfmul.h f5, f0, f1 \n" "vfmul.h f6, f0, f1 \n" "vfmul.h f7, f0, f1 \n"
        "vfmul.h f8, f0, f1 \n" "vfmul.h f9, f0, f1 \n" "vfmul.h f10, f0, f1 \n" "vfmul.h f11, f0, f1 \n"
        "vfmul.h f12, f0, f1 \n" "vfmul.h f13, f0, f1 \n" "vfmul.h f14, f0, f1 \n" "vfmul.h f15, f0, f1 \n"
        "vfmul.h f16, f0, f1 \n" "vfmul.h f17, f0, f1 \n" "vfmul.h f18, f0, f1 \n" "vfmul.h f19, f0, f1 \n"
        "vfadd.h f4, f4, f6 \n" "vfadd.h f5, f5, f7 \n" "vfadd.h f8, f8, f10 \n" "vfadd.h f9, f9, f11 \n"
        "vfadd.h f12, f12, f14 \n" "vfadd.h f13, f13, f15 \n" "vfadd.h f16, f16, f18 \n" "vfadd.h f17, f17, f19 \n"
        "vfadd.h f4, f4, f8 \n" "vfadd.h f5, f5, f9 \n" "vfadd.h f12, f12, f16 \n" "vfadd.h f13, f13, f17 \n"
        "vfadd.h f4, f4, f12 \n" "vfadd.h f5, f5, f13 \n"
        "fmv.d f2, f4 \n" "fmv.d f2, f5 \n"
        "addi %[iters], %[iters], -1 \n" "bnez %[iters], 1b \n"
        : [iters] "+r"(iters) :: "f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","memory"
    );
    snrt_ssr_disable();
}

void opt_quant_fp16(float16_t *in, float16_t *out, const float16_t *recip, uint64_t *magic_p) {
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 16, 8); snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 16, 8); snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)recip);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 16, 8); snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)out);
    snrt_ssr_enable();
    asm volatile(
        "fld     f10, 0(%[magic]) \n"
        "frep.o  %[cnt], 12, 0, 0 \n"    
        "vfmul.h f4, f0, f1 \n" "vfmul.h f5, f0, f1 \n" "vfmul.h f6, f0, f1 \n" "vfmul.h f7, f0, f1 \n"          
        "vfadd.h f8, f4, f10 \n" "vfadd.h f9, f5, f10 \n" "vfadd.h f12, f6, f10 \n" "vfadd.h f13, f7, f10 \n"        
        "vfsub.h f2, f8, f10 \n" "vfsub.h f2, f9, f10 \n" "vfsub.h f2, f12, f10 \n" "vfsub.h f2, f13, f10 \n"        
        : : [cnt] "r"(3), [magic] "r"(magic_p)
        : "f0", "f1", "f2", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f12", "f13", "memory" 
    );
    snrt_fpu_fence(); snrt_ssr_disable();
}

/* =========================================================================
   BACKEND PACKAGING ENGINES (Unpacks float16_t down into discrete integers)
========================================================================= */
void unpack_and_zigzag_diff(const float16_t *in_blocks, int16_t *out_blocks, 
                             int *prev_dc, int num_blocks, const uint8_t *l1_zigzag_map) {
    for (int b = 0; b < num_blocks; b++) {
        const float16_t *in = &in_blocks[b * 64];
        int16_t *out        = &out_blocks[b * 64];

        for (int i = 0; i < 64; i++) {
            uint8_t z_idx = l1_zigzag_map[i];
            
            // 1. Manually NaN-box raw FP16 uint16_t bits for 32/64-bit RISC-V FPU
            uint32_t raw_bits = 0xFFFF0000 | (uint32_t)in[z_idx];
            int32_t val;

            // 2. Load into FPU with fmv.w.x, expand FP16 -> FP32, then convert to int
            asm volatile(
                "fmv.w.x ft0, %[in_bits]     \n\t" // Move NaN-boxed bits into FPU
                "fcvt.s.h ft1, ft0           \n\t" // Convert FP16 -> FP32
                "fcvt.w.s %[out_val], ft1, rtz \n\t" // Truncate FP32 -> INT32
                : [out_val] "=r"(val)
                : [in_bits] "r"(raw_bits)
                : "ft0", "ft1"
            );

            // 3. Process DC vs AC
            if (i == 0) {
                int current_dc = val;
                out[0]         = (int16_t)(current_dc - *prev_dc);
                *prev_dc       = current_dc;
            } else {
                out[i] = (int16_t)val;
            }
        }
    }
}

/* =========================================================================
   HYBRID STAGE 5a: WIDEN FP16 -> FP64 (natural order, contiguous)
   flh loads a single 16-bit lane and NaN-boxes it in hardware, so the
   scalar fcvt.d.h sees a valid half. 4 values in flight per iteration.
   n must be a multiple of 4.
========================================================================= */
void widen_fp16_to_f64(const float16_t *in, double *out, int n) {
    for (int i = 0; i < n; i += 4) {
        asm volatile(
            "flh      f4,   0(%[src]) \n"
            "flh      f5,   2(%[src]) \n"
            "flh      f6,   4(%[src]) \n"
            "flh      f7,   6(%[src]) \n"
            "fcvt.d.h f28,  f4        \n"
            "fcvt.d.h f29,  f5        \n"
            "fcvt.d.h f30,  f6        \n"
            "fcvt.d.h f31,  f7        \n"
            "fsd      f28,  0(%[dst]) \n"
            "fsd      f29,  8(%[dst]) \n"
            "fsd      f30, 16(%[dst]) \n"
            "fsd      f31, 24(%[dst]) \n"
            :: [src] "r"(&in[i]), [dst] "r"(&out[i])
            : "f4", "f5", "f6", "f7", "f28", "f29", "f30", "f31", "memory"
        );
    }
}

/* =========================================================================
   HYBRID STAGE 5b: FP64 ZIG-ZAG & DIFF (same kernel as the FP64 pipeline)
   ISSR fetches in[zigzag_map[i]] into ft0, the FPU converts under frep
   and pushes to the copift queue (t6), the integer core pops the results.
========================================================================= */
void opt_zigzag_and_diff(const double *in_blocks, int16_t *out_blocks,
                         int *prev_dc, int num_blocks,
                         const uint8_t *zigzag_map) {
    for (int b = 0; b < num_blocks; b++) {
        const double *in = &in_blocks[b * 64];
        int16_t *out     = &out_blocks[b * 64];

        snrt_issr_read(SNRT_SSR_DM0, (void*)in, (void*)zigzag_map,
                       64, SNRT_SSR_IDXSIZE_U8);
        snrt_ssr_enable();

        int32_t val;
        asm volatile(
            "mv t6, x0 \n csrsi copift, 0x1 \n"
            "frep.o %[n], 1, 0, 0 \n"
            "fcvt.w.d t6, ft0 \n"
            "mv %[val], t6 \n"
            : [val] "=r"(val) : [n] "r"(63) : "t6", "ft0", "memory"
        );

        int current_dc = val;
        out[0]   = (int16_t)(current_dc - *prev_dc);
        *prev_dc = current_dc;

        for (int i = 1; i < 64; i++) {
            asm volatile("mv %0, t6" : "=r"(val) :: "t6");
            out[i] = (int16_t)val;
        }
        asm volatile("csrci copift, 0x1 \n" ::: "memory");
        snrt_fpu_fence(); snrt_ssr_disable();
    }
}

void huffman_rle_encode_block(int16_t *zigzag_block, BitWriter *writer, int is_chroma) {
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

/* =========================================================================
   MAIN EXECUTIVE PIPELINE WITH MCU-BY-MCU STREAMING
========================================================================= */
int main() {
    uint16_t width  = IMG_WIDTH;  
    uint16_t height = IMG_HEIGHT; 

    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = (uintptr_t)snrt_l1_next();

    volatile uint32_t *shared_size = (volatile uint32_t *)base;

    // --- Memory Assignments ---
    double    *lut         = (double *)   ALIGN_UP_TCDM((uintptr_t)shared_size + sizeof(uint32_t));
	uint8_t *l1_zigzag     = (uint8_t *)   ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));
    uint8_t   *input_idx   = (uint8_t *)  ALIGN_UP_TCDM((uintptr_t)l1_zigzag + 64);
    
    float16_t *opt_Y       = (float16_t *)ALIGN_UP_TCDM((uintptr_t)input_idx + MCU_TOTAL_PIXELS * 3 * sizeof(uint8_t));
    float16_t *opt_CbCr    = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_Y + MCU_TOTAL_PIXELS * sizeof(float16_t));
    float16_t *opt_cb_f    = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_CbCr + MCU_TOTAL_PIXELS * 2 * sizeof(float16_t));
    float16_t *opt_cr_f    = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_cb_f + MCU_CHROMA_PIXELS * sizeof(float16_t));

    float16_t *opt_shift_out=(float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_cr_f + MCU_CHROMA_PIXELS * sizeof(float16_t)); 
    float16_t *opt_tmp_p1  = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_shift_out + 64 * sizeof(float16_t)); 
    float16_t *opt_dct_Y   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_tmp_p1 + 64 * sizeof(float16_t));
    float16_t *opt_dct_Cb  = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_dct_Y + MCU_TOTAL_PIXELS * sizeof(float16_t));
    float16_t *opt_dct_Cr  = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_dct_Cb + MCU_CHROMA_PIXELS * sizeof(float16_t));

    float16_t *opt_q_Y     = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_dct_Cr + MCU_CHROMA_PIXELS * sizeof(float16_t));
    float16_t *opt_q_Cb    = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_q_Y + MCU_TOTAL_PIXELS * sizeof(float16_t));
    float16_t *opt_q_Cr    = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_q_Cb + MCU_CHROMA_PIXELS * sizeof(float16_t));

    int16_t   *zz_diff_Y   = (int16_t *)  ALIGN_UP_TCDM((uintptr_t)opt_q_Cr + MCU_CHROMA_PIXELS * sizeof(float16_t));
    int16_t   *zz_diff_Cb  = (int16_t *)  ALIGN_UP_TCDM((uintptr_t)zz_diff_Y + MCU_TOTAL_PIXELS * sizeof(int16_t));
    int16_t   *zz_diff_Cr  = (int16_t *)  ALIGN_UP_TCDM((uintptr_t)zz_diff_Cb + MCU_CHROMA_PIXELS * sizeof(int16_t));
    uint8_t   *jpeg_output_payload = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)zz_diff_Cr + MCU_CHROMA_PIXELS * sizeof(int16_t));

    float16_t *mat_h       = (float16_t *)ALIGN_UP_TCDM((uintptr_t)jpeg_output_payload + 4096);
    uint64_t  *mat_bcast   = (uint64_t *) ALIGN_UP_TCDM((uintptr_t)mat_h + 16 * sizeof(uint64_t));

    float16_t *recip_luma_h   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)mat_bcast + 64 * sizeof(uint64_t));
    float16_t *recip_chroma_h = (float16_t *)ALIGN_UP_TCDM((uintptr_t)recip_luma_h + 64 * sizeof(float16_t));
    double    *recip_luma_d   = (double *)   ALIGN_UP_TCDM((uintptr_t)recip_chroma_h + 64 * sizeof(float16_t));
    double    *recip_chroma_d = (double *)   ALIGN_UP_TCDM((uintptr_t)recip_luma_d + 64 * sizeof(double));

    uint64_t  *magic_vec_p = (uint64_t *) ALIGN_UP_TCDM((uintptr_t)recip_chroma_d + 64 * sizeof(double));

    // Hybrid: FP64 copies of the quantized blocks (3 KB per MCU)
    double    *q_Y_d       = (double *)   ALIGN_UP_TCDM((uintptr_t)magic_vec_p + sizeof(uint64_t));
    double    *q_Cb_d      = (double *)   ALIGN_UP_TCDM((uintptr_t)q_Y_d  + MCU_TOTAL_PIXELS  * sizeof(double));
    double    *q_Cr_d      = (double *)   ALIGN_UP_TCDM((uintptr_t)q_Cb_d + MCU_CHROMA_PIXELS * sizeof(double));

    uint8_t *l3_write_ptr = (uint8_t *)DRAM_OUT_ADDR;
    uint32_t total_compressed_bytes = 0;

    // ========================================================================
    // ONLY THE DM CORE LOADS THE TABLES INTO L1 VIA DMA
    // ========================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(mat_h, (void*)dct_m, 16 * sizeof(uint64_t)); 
        snrt_dma_start_1d(mat_bcast, (void*)dct_m_bcast, 64 * sizeof(uint64_t));
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
    uint32_t cycles_widen   = 0;
    uint32_t cycles_zigzag  = 0;
    uint32_t cycles_huffman = 0;
    uint32_t t_start, t_end;

    if (snrt_is_compute_core() && core_id == 0) {
        init_standard_ac_tables();
        build_quant_tables(std_luma_q, recip_luma_h, recip_luma_d, JPEG_QUALITY);
        build_quant_tables(std_chroma_q, recip_chroma_h, recip_chroma_d, JPEG_QUALITY);

        float16_t magic_val = float_to_fp16(1536.0f);
        *magic_vec_p = ((uint64_t)magic_val << 48) | ((uint64_t)magic_val << 32) | 
                       ((uint64_t)magic_val << 16) | magic_val;

        for (int i = 0; i < 256; i++) lut[i] = (double)i;
		for (int i = 0; i < 64; i++) l1_zigzag[i] = zigzag_map[i];

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
        for (int i = 0; i < 64; i++) write_byte_out(&writer, std_luma_q[l1_zigzag[i]]);

        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xDB); 
        write_byte_out(&writer, 0x00); write_byte_out(&writer, 0x43); 
        write_byte_out(&writer, 0x01); 
        for (int i = 0; i < 64; i++) write_byte_out(&writer, std_chroma_q[l1_zigzag[i]]);

        // DHT Markers
        write_dht_marker(&writer, std_luma_dc_bits, std_luma_dc_vals, 0x0C, 0x00);   
        write_dht_marker(&writer, luma_ac_bits, luma_ac_vals, 0xA2, 0x10);          
        write_dht_marker(&writer, std_chroma_dc_bits, std_chroma_dc_vals, 0x0C, 0x01); 
        write_dht_marker(&writer, chroma_ac_bits, chroma_ac_vals, 0xA2, 0x11);      

        // SOF0 marker (dynamic dimensions)
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

        *shared_size = writer.byte_pos;
    }

    snrt_cluster_hw_barrier();

    // Transfer the initial headers to L3 DRAM
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l3_write_ptr, jpeg_output_payload, *shared_size);
        snrt_dma_wait_all();
        l3_write_ptr += *shared_size;
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && core_id == 0) {
        total_compressed_bytes += *shared_size;
        writer.byte_pos = 0; 
    }
    snrt_cluster_hw_barrier();

    // ========================================================================
    // MAIN LOOP WITH THE OPTIMIZED FP16 KERNELS (STREAMING ONE MCU AT A TIME)
    // ========================================================================
    for (uint32_t mcu = 0; mcu < total_mcus; mcu++) {
        
        if (snrt_is_dm_core()) {
            int mcu_row = mcu / mcus_wide;
            int mcu_col = mcu % mcus_wide;
            
            uint8_t *dst = input_idx; 
            void *src = (void*)&lena_pixels[(mcu_row * 16 * width + mcu_col * 16) * 3];
            
            snrt_dma_start_2d(dst, src, 16 * 3, 16 * 3, width * 3, 16, 0);
            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core() && core_id == 0) {
            
            printf("Processing MCU %u/%d...\n", mcu + 1, total_mcus);

            // --- STAGE 1: YCbCr Conversion ---
            t_start = snrt_mcycle();
            opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut, MCU_TOTAL_PIXELS);
            t_end = snrt_mcycle();
            cycles_ycbcr += (t_end - t_start);

            // --- STAGE 2: Chroma Subsampling ---
            t_start = snrt_mcycle();
            opt_chroma_420_2d_ssr((double*)opt_CbCr, opt_cb_f, opt_cr_f);
            t_end = snrt_mcycle();
            cycles_chroma += (t_end - t_start);

            // --- STAGE 3: 2D DCT ---
            t_start = snrt_mcycle();
            for(int b = 0; b < 4; b++) {
                simd_dct_pass1(&opt_Y[b*64], opt_shift_out, opt_tmp_p1, mat_h);
                simd_dct_pass2(opt_tmp_p1, &opt_dct_Y[b*64], mat_bcast);
            }
            simd_dct_pass1(opt_cb_f, opt_shift_out, opt_tmp_p1, mat_h);
            simd_dct_pass2(opt_tmp_p1, opt_dct_Cb, mat_bcast);
            simd_dct_pass1(opt_cr_f, opt_shift_out, opt_tmp_p1, mat_h);
            simd_dct_pass2(opt_tmp_p1, opt_dct_Cr, mat_bcast);
            t_end = snrt_mcycle();
            cycles_dct += (t_end - t_start);

            // --- STAGE 4: Quantization ---
            t_start = snrt_mcycle();
            for(int b = 0; b < 4; b++) opt_quant_fp16(&opt_dct_Y[b*64], &opt_q_Y[b*64], recip_luma_h, magic_vec_p);
            opt_quant_fp16(opt_dct_Cb, opt_q_Cb, recip_chroma_h, magic_vec_p);
            opt_quant_fp16(opt_dct_Cr, opt_q_Cr, recip_chroma_h, magic_vec_p);
            t_end = snrt_mcycle();
            cycles_quant += (t_end - t_start);

#if USE_HYBRID_ZIGZAG
            // --- STAGE 5a: Widen FP16 -> FP64 ---
            t_start = snrt_mcycle();
            widen_fp16_to_f64(opt_q_Y,  q_Y_d,  MCU_TOTAL_PIXELS);
            widen_fp16_to_f64(opt_q_Cb, q_Cb_d, MCU_CHROMA_PIXELS);
            widen_fp16_to_f64(opt_q_Cr, q_Cr_d, MCU_CHROMA_PIXELS);
            snrt_fpu_fence();
            t_end = snrt_mcycle();
            cycles_widen += (t_end - t_start);

            // --- STAGE 5b: FP64 Zig-Zag & Diff (ISSR + copift) ---
            t_start = snrt_mcycle();
            opt_zigzag_and_diff(q_Y_d,  zz_diff_Y,  &prev_dc_Y,  4, l1_zigzag);
            opt_zigzag_and_diff(q_Cb_d, zz_diff_Cb, &prev_dc_Cb, 1, l1_zigzag);
            opt_zigzag_and_diff(q_Cr_d, zz_diff_Cr, &prev_dc_Cr, 1, l1_zigzag);
            t_end = snrt_mcycle();
            cycles_zigzag += (t_end - t_start);
#else
            // --- STAGE 5: Zig-Zag Parsing & Quant Unpacking (original FP16) ---
            t_start = snrt_mcycle();
            unpack_and_zigzag_diff(opt_q_Y, zz_diff_Y, &prev_dc_Y, 4, l1_zigzag);
            unpack_and_zigzag_diff(opt_q_Cb, zz_diff_Cb, &prev_dc_Cb, 1, l1_zigzag);
            unpack_and_zigzag_diff(opt_q_Cr, zz_diff_Cr, &prev_dc_Cr, 1, l1_zigzag);
            t_end = snrt_mcycle();
            cycles_zigzag += (t_end - t_start);
#endif

            // --- STAGE 6: Bitstream Pack & Entropy Encoding ---
            t_start = snrt_mcycle();
            for (int b = 0; b < 4; b++) {
                huffman_rle_encode_block(&zz_diff_Y[b * 64], &writer, 0); 
            }
            huffman_rle_encode_block(zz_diff_Cb, &writer, 1); 
            huffman_rle_encode_block(zz_diff_Cr, &writer, 1); 
            t_end = snrt_mcycle();
            cycles_huffman += (t_end - t_start);

            *shared_size = writer.byte_pos;
        }

        snrt_cluster_hw_barrier();

        // [MCU DMA OUT] - Store the bytes of the current MCU in DRAM
        if (snrt_is_dm_core()) {
            if (*shared_size > 0) {
                snrt_dma_start_1d(l3_write_ptr, jpeg_output_payload, *shared_size);
                snrt_dma_wait_all();
                l3_write_ptr += *shared_size;
            }
        }
        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core() && core_id == 0) {
            total_compressed_bytes += *shared_size;
            writer.byte_pos = 0; 
        }
        snrt_cluster_hw_barrier();
    }

    // ========================================================================
    // CLOSING PHASE (POST-LOOP)
    // ========================================================================
    if (snrt_is_compute_core() && core_id == 0) {
        flush_bit_writer(&writer); 
        write_byte_out(&writer, 0xFF); write_byte_out(&writer, 0xD9); 
        *shared_size = writer.byte_pos;
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        if (*shared_size > 0) {
            snrt_dma_start_1d(l3_write_ptr, jpeg_output_payload, *shared_size);
            snrt_dma_wait_all();
        }
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && core_id == 0) {
        total_compressed_bytes += *shared_size;
    }
    snrt_cluster_hw_barrier();

    // ========================================================================
    // POST-LOOP PHASE: PRINT RESULTS FROM L3 DRAM
    // ========================================================================
    if (snrt_is_compute_core() && core_id == 0) {
        uint8_t *l3_output = (uint8_t *)DRAM_OUT_ADDR;
        
#if USE_HYBRID_ZIGZAG
        printf("\n=========  PERFORMANCE PROFILE (HYBRID FP16 + FP64 ZIG-ZAG)  =========\n");
#else
        printf("\n================  PERFORMANCE PROFILE (HARDWARE FP16)  ===============\n");
#endif
        printf("  Total MCUs Processed      : %u\n", total_mcus);
        printf("  Stage 1: YCbCr Conversion : %u cycles\n", cycles_ycbcr);
        printf("  Stage 2: Subsampling 4:2:0: %u cycles\n", cycles_chroma);
        printf("  Stage 3: Forward 2D-DCT   : %u cycles\n", cycles_dct);
        printf("  Stage 4: Quantization     : %u cycles\n", cycles_quant);
#if USE_HYBRID_ZIGZAG
        printf("  Stage 5a: Widen FP16->FP64: %u cycles\n", cycles_widen);
        printf("  Stage 5b: ZigZag & Diff   : %u cycles\n", cycles_zigzag);
#else
        printf("  Stage 5: ZigZag & Diff    : %u cycles\n", cycles_zigzag);
#endif
        printf("  Stage 6: Huffman Encoding : %u cycles\n", cycles_huffman);
        printf("------------------------------------------------------\n");
        printf("  Total Compute Pipeline    : %u cycles\n", 
               cycles_ycbcr + cycles_chroma + cycles_dct + cycles_quant + cycles_widen + cycles_zigzag + cycles_huffman);
        printf("  Per MCU                   : %u cycles\n",
               (cycles_ycbcr + cycles_chroma + cycles_dct + cycles_quant + cycles_widen + cycles_zigzag + cycles_huffman) / total_mcus);
        printf("=====================================================\n\n");
        
        printf("  Final Valid JPG File Size  : %u bytes\n", total_compressed_bytes);
        printf("=====================================================\n");
        
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
