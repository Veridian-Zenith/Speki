// SPDX-License-Identifier: OSL-3.0
// raw_alloc.h — arena allocator via raw mmap/munmap/madvise/mlock.
//
// Speki's only allocator for short-lived and long-lived objects is this arena.
// No malloc, no free. We mmap regions once, bump a pointer, never free
// individual allocations during a request lifecycle.
//
// For long-lived tensors (the model weights), we use a separate "tier-2"
// allocator backed by mmap+mlock so the OS can't swap the model out.

#ifndef SPEKI_RAW_ALLOC_H
#define SPEKI_RAW_ALLOC_H

#include "raw_syscalls.h"

// NULL — libc equivalent. We don't depend on stddef.h.
#ifndef NULL
#define NULL ((void*)0)
#endif

// MAP_NORESERVE (kernel ABI constant) — don't reserve swap space for the
// mapping. Most user mappings don't need it.
#define MAP_NORESERVE     0x00004000

// ─── Arena: bump allocator over a single mmap region ─────────────────────────
//
// Usage:
//   Arena a = arena_create(1 << 20); // 1 MiB
//   void* p = arena_alloc(a, 64, 8);  // 64 bytes, 8-byte aligned
//   arena_reset(a);                  // reuse from the top, no unmaps
//   arena_destroy(a);                // munmap + free the region
//
// Thread-safety: NOT thread-safe. One arena per thread, or external sync.
//
// Determinism: arenas are deterministic in address layout (modulo kernel
// ASLR of the base). For determinism we always pass MAP_FIXED-style hints
// or accept the ASLR randomization on the base address only.

typedef struct Arena {
    u8*  base;     // mmap'd region
    usize cap;     // total bytes reserved
    usize off;     // next allocation offset
    u32  flags;    // MAP_PRIVATE | MAP_ANONYMOUS, etc.
} Arena;

// Reserved region size defaults.
#define ARENA_DEFAULT_CAP    (1 << 20)   // 1 MiB default
#define ARENA_MIN_CAP        (1 << 12)   // 4 KiB page
#define ARENA_ALIGN_DEFAULT  16          // AVX2 wants 16+ byte alignment
#define ARENA_ALIGN_SIMD     32          // AVX-512 wants 32+
#define ARENA_ALIGN_PAGE     4096        // page-aligned

static inline Arena arena_create(usize cap) {
    Arena a = {0};
    if (cap < ARENA_MIN_CAP) cap = ARENA_MIN_CAP;

    u8* p = (u8*)raw_mmap((void*)0, cap,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                          -1, 0);
    // raw_mmap returns (void*)-1 (i.e. 0xffffffffffffffff) on failure.
    if (p == (u8*)(usize)-1) return a;

    a.base  = p;
    a.cap   = cap;
    a.off   = 0;
    a.flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    return a;
}

// arena_alloc(a, n, align) → aligned pointer or NULL if out of space.
//
// Alignment must be a power of two. We round up `n` to the alignment
// boundary to keep subsequent allocations aligned.
static inline void* arena_alloc(Arena* a, usize n, usize align) {
    if (!a || !a->base) return (void*)0;

    // Compute aligned offset: round up to `align`.
    usize aligned_off = (a->off + (align - 1)) & ~(align - 1);

    // Check overflow + capacity.
    if (aligned_off > a->cap || n > a->cap - aligned_off) {
        return (void*)0;  // out of space; could grow via mremap, but we don't
    }

    void* p = a->base + aligned_off;
    a->off = aligned_off + n;

    // Zero-init per the project's zero-init policy (matches your fish CFLAGS).
    // Use a tight inline loop instead of memset (no libc dependency).
    u8* b = (u8*)p;
    for (usize i = 0; i < n; i++) b[i] = 0;
    return p;
}

// arena_reset(a) — rewind the bump pointer. The pages are still mapped,
// so subsequent allocations are fast. Does NOT zero memory; caller's
// responsibility to overwrite before reuse if they need zero-init.
static inline void arena_reset(Arena* a) {
    if (a) a->off = 0;
}

// arena_destroy(a) — munmap the region. After this, the arena is dead.
static inline void arena_destroy(Arena* a) {
    if (a && a->base && a->cap > 0) {
        (void)raw_munmap(a->base, a->cap);
    }
    a->base = (u8*)0;
    a->cap = 0;
    a->off = 0;
}

// arena_used(a) → bytes currently allocated
static inline usize arena_used(const Arena* a) {
    return a ? a->off : 0;
}

// arena_remaining(a) → bytes left before exhaustion
static inline usize arena_remaining(const Arena* a) {
    return a ? (a->cap - a->off) : 0;
}

#endif // SPEKI_RAW_ALLOC_H