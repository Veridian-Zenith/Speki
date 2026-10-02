// SPDX-License-Identifier: OSL-3.0
// f16c.h — FP16 ↔ FP32 conversion via F16C intrinsics.
//
// Alder Lake exposes F16C (a.k.a. CVT16) which does half↔single conversion
// at one conversion per cycle per lane. We use it to promote FP16 weights
// to FP32 for our matmul accumulator and to demote FP32 activations to
// FP16 for storage.
//
// <immintrin.h> is provided by clang's resource-dir and is header-only —
// it doesn't pull in any libc. We include it directly.

#ifndef SPEKI_F16C_H
#define SPEKI_F16C_H

#include "../crt/raw_syscalls.h"
#include "types.h"

#include <immintrin.h>

// ─── Scalar conversions (FP16 <-> FP32) ────────────────────────────────────

// f16_to_f32_bits(h) — bit-cast FP16 to FP32 by promoting through F16C.
//
// _cvtsh_ss takes the 16 bits AS AN unsigned short and returns the fp32
// value. Do not insert a (__fp16) cast in between: that is a VALUE conversion
// from integer to half, not a bit reinterpretation, so 0x4248 (whose bits
// encode 3.140625) would first become the half nearest to the integer 16968,
// i.e. inf, and the result came back as 3.125 instead. Every fp16 -> fp32
// promotion in the project was affected by this.
static inline fp32 f16_to_f32(fp16 h) {
    return _cvtsh_ss((unsigned short)h);
}

// f32_to_f16(f) — demote FP32 to FP16 via F16C.
//
// _cvtss_sh's second argument is an IMMEDIATE, not a runtime value: the low
// bits select the rounding mode and the upper bits suppress exceptions. So
// this cannot be written as a plain helper taking a mode parameter -- the mode
// has to be a compile-time constant.
//
// The mode here must be _MM_FROUND_TO_NEAREST_INT (0x00), which selects
// round-to-nearest-even. Passing 0 as a *bare literal* looks identical but is
// not the same thing to reason about, and more importantly the default for
// this intrinsic when rounding bits read as "use MXCSR" is to TRUNCATE. That
// silently produced 0x4248 (= 3.125) for pi, where correctly-rounded is
// 0x4249 (= 3.140625) -- an error of 0.0166, not a rounding artefact.
// Values that are exactly representable in fp16 were unaffected, which is why
// the bug survived: every other f16c test in the tree used such values.
static inline fp16 f32_to_f16(fp32 f) {
    return (fp16)(u16)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | 0x00);
}

// ─── SIMD conversions (8x or 16x in one instruction) ──────────────────────

// f16x8_to_f32x8: convert 8 packed FP16 values to 8 FP32 values.
//
// `src` is a pointer to 8 contiguous FP16 (16 bytes total).
// `dst` is a pointer to 8 contiguous FP32 (32 bytes total).
static inline void f16x8_to_f32x8(const fp16* src, fp32* dst) {
    __m128i h = _mm_loadu_si128((const __m128i*)src);
    __m256  f = _mm256_cvtph_ps(h);
    _mm256_storeu_ps(dst, f);
}

// f32x8_to_f16x8: convert 8 packed FP32 values to 8 FP16 values (packed).
//
// `src` is 8 contiguous FP32 (32 bytes).
// `dst` is 8 contiguous FP16 (16 bytes).
static inline void f32x8_to_f16x8(const fp32* src, fp16* dst) {
    __m256  f = _mm256_loadu_ps(src);
    __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*)dst, h);
}

// f16x16_to_f32x16: convert 16 packed FP16 (32 bytes) to 16 FP32 (64 bytes).
//
// Two f16x8 conversions in sequence.
static inline void f16x16_to_f32x16(const fp16* src, fp32* dst) {
    f16x8_to_f32x8(src,      dst);
    f16x8_to_f32x8(src + 8,  dst + 8);
}

// f32x16_to_f16x16: convert 16 packed FP32 (64 bytes) to 16 FP16 (32 bytes).
static inline void f32x16_to_f16x16(const fp32* src, fp16* dst) {
    f32x8_to_f16x8(src,      dst);
    f32x8_to_f16x8(src + 8,  dst + 8);
}

#endif // SPEKI_F16C_H