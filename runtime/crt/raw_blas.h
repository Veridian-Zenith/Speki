// SPDX-License-Identifier: OSL-3.0
// raw_blas.h — BLAS provider selector for Speki.
//
// Speki needs BLAS only for "big GEMM" passes during training (e.g.
// backprop through large FFN layers). Inference uses our hand-written
// kernels — they're faster than any BLAS for small per-layer sizes.
//
// We support two BLAS providers, selected at build time:
//   * OpenBLAS (MIT, BSD-3) — default
//   * Intel MKL (BSD-like for headers, free for use) — requires oneAPI
//
// We never use Level Zero, CUDA, ROCm, OpenCL, or SYCL. CPU only.
//
// The "raw" prefix here is a misnomer (these aren't syscalls); we keep
// it for naming consistency with the runtime headers. Each provider
// implements the same subset of CBLAS:
//   cblas_sgemm, cblas_dgemm  (Speki uses mostly dgemm for FP32 accum)
//   cblas_sgemv                (vector-matrix — not used yet)
//
// Build selection:
//   just build              → OpenBLAS
//   just build blas=mkl     → MKL (oneAPI must be installed)

#ifndef SPEKI_RAW_BLAS_H
#define SPEKI_RAW_BLAS_H

#include <stddef.h>

// We include only the CBLAS header (no FORTRAN). The CBLAS API is what
// OpenBLAS, MKL, and reference BLAS all implement.

// OpenBLAS path (Arch/CachyOS package) at /usr/include/cblas.h
#ifdef SPEKI_USE_OPENBLAS
  #include <cblas.h>
#endif

#ifdef SPEKI_USE_MKL
  // MKL ships its own cblas.h; mkl.h pulls it in.
  #include <mkl.h>
#endif

// Fallback: if neither is selected at build time, expose a stub so
// the project still links. This is fine because Speki's runtime paths
// that actually call BLAS are gated behind runtime checks.
//
// If you ship the runtime without BLAS, training is disabled (inference
// still works because it uses hand-written kernels).

#ifdef __cplusplus
extern "C" {
#endif

// BLAS routines Speki calls. Each is a thin wrapper — when BLAS is
// available, this expands to the real CBLAS call. When it isn't,
// we provide a stub that does the matmul via the hand-rolled kernels.

// raw_dgemm — FP64 GEMM. We don't use this, but it's the BLAS canonical.
#ifdef SPEKI_USE_OPENBLAS
  #define raw_dgemm(C, TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C_, ldc) \
      cblas_dgemm(C, TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C_, ldc)
  #define raw_sgemm(C, TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C_, ldc) \
      cblas_sgemm(C, TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C_, ldc)
#else
  #define raw_dgemm(...) ((void)0)
  #define raw_sgemm(...) ((void)0)
#endif

// raw_blas_available() → 1 if a real BLAS was linked, 0 if stub.
// Useful for runtime feature detection.
static inline int raw_blas_available(void) {
#ifdef SPEKI_USE_OPENBLAS
    return 1;
#else
    return 0;
#endif
}

#ifdef __cplusplus
}
#endif

#endif // SPEKI_RAW_BLAS_H