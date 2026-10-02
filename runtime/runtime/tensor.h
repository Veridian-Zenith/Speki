// SPDX-License-Identifier: OSL-3.0
// tensor.h — Speki tensor type: FP16 storage, FP32 compute, mmap+mlock.
//
// Speki tensors are mmap'd, page-aligned, optionally hugepage-backed, and
// (for long-lived tensors like model weights) mlock'd into RAM so the OS
// cannot swap them out.
//
// dtype:
//   DT_F32  — FP32 storage, FP32 compute
//   DT_F16  — FP16 storage, FP16→FP32 promotion in compute via F16C
//   DT_I8   — INT8 storage, per-channel FP32 scales (for quantization)
//   DT_I32  — INT32 storage (e.g. token IDs, indices)
//
// layout:
//   ROW_MAJOR    — A[i,j,k,...] = base + i*stride_d * sizes + ...
//   COL_MAJOR    — A[i,j,k,...] = base + (j + i*D2 + k*D2*D1 + ...)
// We support row-major only for now; column-major can come later if needed.
//
// alignment:
//   Each tensor's base address is aligned to at least 32 bytes (AVX2 width),
//   and the leading dimensions are padded so that row strides are multiples
//   of 32 bytes. This matters a lot for SIMD loads in our matmul kernels.

#ifndef SPEKI_TENSOR_H
#define SPEKI_TENSOR_H

#include "../crt/raw_syscalls.h"
#include "../crt/raw_alloc.h"
#include "../crt/raw_io.h"
#include "types.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

// ─── dtype enum ─────────────────────────────────────────────────────────────
typedef enum DType {
    DT_F32  = 0,
    DT_F16  = 1,
    DT_I8   = 2,
    DT_I32  = 3,
} DType;

static inline u32 dtype_size(DType d) {
    switch (d) {
        case DT_F32: return 4;
        case DT_F16: return 2;
        case DT_I8:  return 1;
        case DT_I32: return 4;
        default:     return 0;
    }
}

// ─── shape ──────────────────────────────────────────────────────────────────
//
// Up to SPEKI_RANK_MAX dimensions. Most tensors we'll see in a transformer
// are 1D, 2D, or 3D (batch, seq, hidden). 4D is rare (multi-head attention
// reshapes). Anything beyond 4D goes through the generic path.
#define SPEKI_RANK_MAX 8

typedef struct Shape {
    u32 n;          // number of dimensions (rank)
    u32 d[SPEKI_RANK_MAX];   // dimension sizes
} Shape;

static inline usize shape_numel(const Shape* s) {
    usize n = 1;
    for (u32 i = 0; i < s->n; i++) n *= (usize)s->d[i];
    return n;
}

// shape_byte_size(s, dtype) → total bytes for the underlying buffer.
static inline usize shape_byte_size(const Shape* s, DType dt) {
    return shape_numel(s) * (usize)dtype_size(dt);
}

// ─── Tensor ─────────────────────────────────────────────────────────────────
//
// data      — base address (mmap'd or arena-allocated, 32-byte aligned)
// byte_cap  — total bytes in the underlying allocation
// shape     — logical dimensions
// stride_b  — row stride in BYTES, padded up so row[i] is 32-byte aligned
// rank      — cached shape.n for fast access
// dtype     — element type
// mlocked   — whether data is mlocked (never swapped) — only meaningful
//             for tensors backed by mmap, not arena
//
// `data` may be NULL for a "view" tensor (slice of another tensor). When
// non-NULL, `data` is a stable address that survives until tensor_destroy.
typedef struct Tensor {
    u8*    data;
    usize  byte_cap;
    Shape  shape;
    usize  stride_b;       // row stride in bytes (>= numel_per_row * dtype_size)
    u32    rank;
    DType  dtype;
    u32    mlocked;        // 1 if mlock'd; 0 otherwise
} Tensor;

#define TENSOR_EMPTY ((Tensor){0})

// ─── Constructors ───────────────────────────────────────────────────────────
//
// tensor_alloc: allocate a tensor in an arena. The base address is taken
// from the arena; we don't mmap here. Useful for activations and scratch
// buffers that live for a single forward pass.
//
// Alignment: 32 bytes (AVX2 width). Stride is padded so each row starts
// on a 32-byte boundary.

static inline Tensor tensor_alloc(Arena* a, Shape s, DType dt) {
    Tensor t = TENSOR_EMPTY;
    if (s.n < 1 || s.n > SPEKI_RANK_MAX) return t;

    usize elem_size = (usize)dtype_size(dt);
    usize numel     = shape_numel(&s);
    usize total_b   = numel * elem_size;

    // Row padding: stride must be a multiple of 32 bytes so any 2D access
    // A[i, j] = data + i*stride_b + j*elem_size is properly aligned.
    if (s.n >= 2) {
        usize row_elems = (usize)s.d[s.n - 1];
        usize row_bytes = row_elems * elem_size;
        t.stride_b = (row_bytes + 31u) & ~((usize)31u);
        total_b = ((usize)s.d[0]) * t.stride_b;
        // For higher-rank tensors, we'd need to be more careful, but
        // transformer tensors are typically ≤ 3D. Pad only outer dim.
        for (u32 i = 1; i + 1 < s.n; i++) total_b *= (usize)s.d[i];
    } else {
        t.stride_b = (total_b + 31u) & ~((usize)31u);
    }

    t.data = (u8*)arena_alloc(a, total_b, 32);
    if (!t.data) return t;

    t.byte_cap = total_b;
    t.shape    = s;
    t.rank     = s.n;
    t.dtype    = dt;
    t.mlocked  = 0;
    return t;
}

// tensor_load: mmap a file from disk and wrap it as a tensor. Optionally
// mlock it into RAM.
//
// The file at `path` is mapped read-only (or read-write if RW=1). Size is
// determined from the file size via raw_lseek(SEEK_END). We always map
// exactly the file size rounded up to a page boundary (so the trailing
// partial page is zero-padded by the kernel).
//
// For INT8 weight files, we expect a small header: the layout is documented
// in tools/quantize.rs — for now we treat the file as raw element data.
static inline Tensor tensor_load(const char* path, Shape s, DType dt, int rw) {
    Tensor t = TENSOR_EMPTY;
    if (s.n < 1 || s.n > SPEKI_RANK_MAX) return t;

    i32 flags = O_RDONLY_RAW | O_CLOEXEC_RAW;
    if (rw) flags = O_RDWR_RAW | O_CLOEXEC_RAW;

    i32 fd = raw_openat(AT_FDCWD_RAW, path, flags, 0);
    if (fd < 0) return t;

    // Get file size.
    slong end = raw_lseek(fd, 0, 2);   // SEEK_END
    if (end < 0) { raw_close(fd); return t; }

    usize file_size = (usize)end;
    usize elem_size = (usize)dtype_size(dt);
    usize want      = shape_byte_size(&s, dt);

    if (file_size < want) {
        raw_close(fd);
        return t;  // file too small for shape
    }

    // Round up to page size for the mmap length.
    usize page_size = 4096;
    usize map_size  = (file_size + page_size - 1) & ~(page_size - 1);

    i32 prot = rw ? (PROT_READ | PROT_WRITE) : PROT_READ;

    void* p = raw_mmap(NULL, map_size, prot, MAP_PRIVATE, fd, 0);
    if ((u8*)p == (u8*)(usize)-1) { raw_close(fd); return t; }

    // Compute stride for 2D+ tensors.
    usize stride_b;
    if (s.n >= 2) {
        usize row_elems = (usize)s.d[s.n - 1];
        usize row_bytes = row_elems * elem_size;
        stride_b = (row_bytes + 31u) & ~((usize)31u);
    } else {
        stride_b = (file_size + 31u) & ~((usize)31u);
    }

    t.data     = (u8*)p;
    t.byte_cap = file_size;
    t.shape    = s;
    t.stride_b = stride_b;
    t.rank     = s.n;
    t.dtype    = dt;
    t.mlocked  = 0;

    return t;
}

// tensor_mlock: lock the tensor's pages into RAM. After this call, the OS
// will never swap these pages out (they remain in physical memory until
// tensor_munlock or process exit). Use for model weights only.
//
// Returns 0 on success, -errno on failure. Typical failure is ENOMEM
// (rlimit-locked memory exceeded; we should not pin more than ~25% of RAM).
static inline i32 tensor_mlock(Tensor* t) {
    if (!t || !t->data || t->mlocked) return 0;
    i32 r = raw_mlock(t->data, t->byte_cap);
    if (r == 0) t->mlocked = 1;
    return r;
}

// tensor_munmap: unmap and close any underlying fd. After this the tensor
// is dead (set to TENSOR_EMPTY). For arena tensors, this is a no-op for
// the unmap; the arena memory is reclaimed at arena_destroy.
static inline void tensor_munmap(Tensor* t) {
    if (!t) return;
    if (t->data) {
        (void)raw_munmap(t->data, t->byte_cap);
    }
    t->data = (u8*)0;
    t->byte_cap = 0;
    t->shape = (Shape){0};
    t->stride_b = 0;
    t->rank = 0;
    t->dtype = DT_F32;
    t->mlocked = 0;
}

// tensor_madvise_hugepage: hint that this tensor should use hugepages.
// Useful for large weight tensors that benefit from 2 MiB pages.
//
// Returns 0 on success, -errno on failure. Common failure is EINVAL if
// the system doesn't have hugepage support configured.
static inline i32 tensor_madvise_hugepage(Tensor* t) {
    if (!t || !t->data) return 0;
    return raw_madvise(t->data, t->byte_cap, MADV_HUGEPAGE);
}

// ─── Element access (no bounds checking — kernel ABI style) ────────────────
//
// tensor_ptr(t, indices...) returns a void* to the element at the given
// logical coordinates. The compiler may inline and constant-fold this when
// indices are known at compile time.
//
// We don't check rank-mismatched argument counts; the caller is responsible
// for matching t->rank. Variadic with up to SPEKI_RANK_MAX indices.

// Helper: linear offset in elements from a contiguous array of indices.
static inline usize tensor_offset(const Tensor* t, const u32* idx) {
    usize off = 0;
    // For each leading dim i (except the last), accumulate.
    usize stride = 1;
    for (i32 i = (i32)t->rank - 1; i >= 0; i--) {
        // Stride in ELEMENTS for dim i.
        usize dim_stride = 1;
        for (u32 j = (u32)(i + 1); j < t->rank; j++) {
            dim_stride *= (usize)t->shape.d[j];
        }
        off += (usize)idx[i] * dim_stride;
    }
    (void)stride; // unused; kept for clarity
    return off;
}

// tensor_ptr(t, idx0, idx1, ...): return a typed pointer to the element.
// Defined via overload-style macro: pick the right one for the rank.
// For ranks we haven't specialized, fall back to tensor_offset + raw_ptr.

#define TENSOR_AT(T, ...) (&((T*)((T).data))[tensor_offset(&(T), ((u32[]){__VA_ARGS__}))])

#endif // SPEKI_TENSOR_H