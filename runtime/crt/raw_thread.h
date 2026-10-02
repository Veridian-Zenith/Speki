// SPDX-License-Identifier: OSL-3.0
// raw_thread.h — raw clone() + futex() — no pthreads.
//
// Threads are created via clone(2) with CLONE_VM | CLONE_FS | CLONE_FILES
// | CLONE_SIGHAND | CLONE_THREAD (the Linux "thread" set) and given a
// freshly mmap'd stack. Synchronization is futex(2) only.
//
// We do NOT use pthread_self() — instead every Thread struct holds a tid
// returned by clone() (which on Linux is the kernel thread ID).

#ifndef SPEKI_RAW_THREAD_H
#define SPEKI_RAW_THREAD_H

#include "raw_syscalls.h"
#include "raw_alloc.h"

// clone() flags — the subset Speki uses.
// We use a "thread" clone: shares address space, file descriptors, signal
// handlers, etc. Each thread still gets its own tid and kernel scheduling.
#define CLONE_VM             0x00000100
#define CLONE_FS             0x00000200
#define CLONE_FILES          0x00000400
#define CLONE_SIGHAND        0x00000800
#define CLONE_THREAD         0x00010000
#define CLONE_DETACHED       0x00400000  // no join needed
#define CLONE_IO             0x80000000

// futex ops
#define FUTEX_WAIT           0
#define FUTEX_WAKE           1
#define FUTEX_WAIT_PRIVATE 128
#define FUTEX_WAKE_PRIVATE 129

// futex() syscall: raw_futex(addr, op, val, timeout, addr2, val3)
static inline slong raw_futex(u32* addr, i32 op, u32 val,
                              const struct timespec* timeout,
                              u32* addr2, u32 val3) {
    return SPEKI_SYSCALL6(SYS_futex, addr, op, val, timeout, addr2, val3);
}

// futex_wait(addr, expected) — atomic wait. If *addr == expected, block
// until FUTEX_WAKE changes it. Always uses PRIVATE for kernel-futex
// locality (this is a process-local futex, not shared).
static inline i32 raw_futex_wait(u32* addr, u32 expected) {
    return (i32)raw_futex(addr, FUTEX_WAIT_PRIVATE, expected,
                          (struct timespec*)0, (u32*)0, 0);
}

// futex_wake(addr, n) — wake up to `n` threads blocked on *addr.
// Returns the number of woken threads, or -errno.
static inline i32 raw_futex_wake(u32* addr, u32 n) {
    return (i32)raw_futex(addr, FUTEX_WAKE_PRIVATE, n,
                          (struct timespec*)0, (u32*)0, 0);
}

// ─── Futex-based mutex ─────────────────────────────────────────────────────
//
// Layout (8 bytes, naturally aligned on x86_64):
//   bits 0..30: owner tid (0 = unlocked)
//   bit 31:     lock flag
//
// We use a simple state machine:
//   state == 0                → unlocked
//   state == 1                → locked, no waiters
//   state == 2                → locked, with waiters (futex-wait mode)
//
// Atomic via __atomic_* builtins (LLVM intrinsics; map to x86 LOCK prefix).

typedef struct Mutex {
    u32 state;
} Mutex;

#define MUTEX_INIT {0}

// Mutex primitives. We don't track owner tid for simplicity — caller
// is responsible for not using the same Mutex from the holder thread
// recursively.
static inline void mutex_lock(Mutex* m) {
    u32 zero = 0;
    if (__atomic_compare_exchange_n(&m->state, &zero, 1, 0,
                                    __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        return;  // fast path: uncontended
    }
    // Slow path: mark as contended and futex-wait.
    while (1) {
        u32 s = __atomic_load_n(&m->state, __ATOMIC_RELAXED);
        if (s == 0) {
            // Try to take it now.
            zero = 0;
            if (__atomic_compare_exchange_n(&m->state, &zero, 1, 0,
                                            __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
                return;
            }
            continue;
        }
        // Transition to contended.
        u32 contended = 2;
        if (s == 1 && __atomic_compare_exchange_n(&m->state, &s, contended, 0,
                                                  __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            // Block until the state changes (typically to 0 after unlock).
            (void)raw_futex_wait(&m->state, contended);
        }
        // Loop and try to acquire again.
    }
}

static inline void mutex_unlock(Mutex* m) {
    u32 s = __atomic_load_n(&m->state, __ATOMIC_RELAXED);
    // If there are waiters (state == 2), wake one.
    if (s == 2) {
        u32 one = 1;
        if (__atomic_compare_exchange_n(&m->state, &one, 0, 0,
                                        __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
            (void)raw_futex_wake(&m->state, 1);
            return;
        }
    }
    // No waiters or already cleared: just store 0.
    __atomic_store_n(&m->state, 0, __ATOMIC_RELEASE);
}

// ─── Thread ─────────────────────────────────────────────────────────────────
//
// Thread is created with clone() and runs the user function. The function
// signature is `int (*)(void*)` returning an int (matches pthread style).
// We don't pass it through pthread_create's pthread_t — instead we track
// the kernel tid returned by clone().

// raw_clone(fn, arg, stack, flags) → kernel tid of new thread, or -errno.
// `fn` receives `arg`; the int return value is ignored (use a shared arena
// or atomic for results).
//
// `stack` must point to the TOP of a stack region (the kernel grows down).
// `stack_size` is the size of the region we mmap'd for this thread.

typedef int (*ThreadFn)(void*);

// Forward declaration — actual definition is in raw_thread.c.
// We split out the body because the inline asm for clone() is non-trivial.
extern slong raw_clone(ThreadFn fn, void* arg, void* stack_top, u32 flags);

#endif // SPEKI_RAW_THREAD_H