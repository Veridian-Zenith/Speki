// SPDX-License-Identifier: OSL-3.0
// crt0.c — Speki entry point.
//
// _start is written in inline asm to set up the stack exactly the way the
// AMD64 ABI expects. After _start, control transfers to speki_main with
// rsp 16-byte aligned per ABI convention.
//
// Why inline asm: clang's prologue for `void _start(void)` assumes rsp is
// 8-byte aligned (the state after a `call`). But the Linux kernel hands
// us rsp 16-byte aligned. If we let clang emit its prologue, the combined
// pushq/subq sequence lands rsp 8-aligned, and any aligned SSE op
// (vmovdqa, vmovaps) inside inlined functions faults with #GP.
//
// Built with -nostdlib. No libc, no libgcc, no compiler-rt. We provide our
// own __stack_chk_fail, memset, and memcpy in crt_extras.c.

#include "raw_syscalls.h"
#include "raw_time.h"
#include "raw_io.h"
#include "raw_log.h"
#include "raw_alloc.h"

// Forward declarations for runtime tests. Each runtime/*.c file with a
// test entry point declares its function here so speki_main can call it.
extern int tensor_test_main(void);
extern int kernels_test_main(void);

// speki_main — the real entry point, called by _start.
//
// This function runs with rsp 8-byte aligned at function entry (the ABI
// state), so clang's prologue (pushq %rbx + subq) lands rsp 16-aligned
// and all SSE ops inside the body are correctly aligned.
__attribute__((noinline, noreturn, used))
void speki_main(void) {
    u64 t0 = raw_now_ns();

    Arena scratch = arena_create(1 << 16); // 64 KiB scratch arena
    if (scratch.base != (u8*)(usize)-1 && scratch.base) {
        u64* slots = (u64*)arena_alloc(&scratch, 8 * sizeof(u64), 16);
        if (slots) {
            for (u32 i = 0; i < 8; i++) slots[i] = 0xA5A5A5A5A5A5A5A5ull + i;
        }
    }

    SPEKI_LOG(LOG_INFO, "speki: ok");

    u64 t1 = raw_now_ns();
    u64 elapsed_ns = (t1 > t0) ? (t1 - t0) : 0;

    char msg[64];
    usize off = 0;
    {
        const char prefix[] = "boot_ns=";
        for (usize k = 0; k < sizeof(prefix) - 1 && off < sizeof(msg); k++) {
            msg[off++] = prefix[k];
        }
        if (elapsed_ns == 0) {
            msg[off++] = '0';
        } else {
            char tmp[24];
            usize j = 0;
            u64 v = elapsed_ns;
            while (v > 0 && j < sizeof(tmp)) {
                tmp[j++] = (char)('0' + (v % 10));
                v /= 10;
            }
            while (j > 0 && off < sizeof(msg)) msg[off++] = tmp[--j];
        }
    }
    SPEKI_LOG_BUF(LOG_DEBUG, msg, off);

    arena_destroy(&scratch);

    // Run runtime tests. If any fail, exit non-zero so CI catches it.
    int test_rc = tensor_test_main();
    test_rc |= kernels_test_main();
    if (test_rc != 0) {
        SPEKI_LOG(LOG_CRIT, "tests failed");
    }
    (void)test_rc;

    raw_exit_group(0);
    __builtin_unreachable();
}

// _start — hand-written, never touched by clang's prologue generator.
//
// The Linux kernel gives us:
//   - rsp: pointing at argc/argv/envp stack, 16-byte aligned
//   - rdx: address of fini (if provided by ld.so, which we don't use)
//
// We make rsp look like a normal post-`call` frame: subtract 8 to account
// for the missing return-address push. Then we jump into speki_main.
//
// The `jmp` (not `call`) ensures we don't push another return address;
// speki_main never returns anyway.
__attribute__((noreturn, visibility("default")))
void _start(void) {
    __asm__ volatile (
        "sub $8, %%rsp\n"      // mimic post-`call` state: rsp is now 8-aligned
        "jmp speki_main\n"
        ::: "memory"
    );
    __builtin_unreachable();
}