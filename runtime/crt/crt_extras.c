// SPDX-License-Identifier: OSL-3.0
// crt_extras.c — Speki CRT stubs and shared data.
//
// With -nostdlib we must supply any compiler-generated runtime hooks
// ourselves. So far the runtime needs:
//
//   __stack_chk_fail    — called when a stack-canary mismatch is detected
//   memset              — clang emits this for stack-init at -O0
//   memcpy              — clang emits this for struct copies
//
// We deliberately do NOT link libgcc, libunwind, or compiler-rt.
//
// We also define a few small RO symbols that need a stable address across
// translation units (e.g. log level labels) so the linker can dedupe them.

#include "raw_syscalls.h"
#include "raw_log.h"

// Level label table — single rodata instance, shared by every TU.
const char LOG_LEVEL_LABELS[8][5] = {
    "EMRG", "ALRT", "CRIT", "ERR ", "WARN", "NOTE", "INFO", "DEBG"
};

// raw_memset(dst, value, len) — like libc memset, but no libc.
// Hand-rolled byte fill; the compiler will optimize this when len is constant.
__attribute__((no_sanitize("address")))
void* memset(void* dst, int value, unsigned long len) {
    unsigned char* p = (unsigned char*)dst;
    for (unsigned long i = 0; i < len; i++) p[i] = (unsigned char)value;
    return dst;
}

// raw_memcpy(dst, src, len) — like libc memcpy, but no libc.
// Non-overlapping copy only; clang never emits overlapping memcpy for stack
// variables (that would be UB), so we don't handle the overlap case here.
__attribute__((no_sanitize("address")))
void* memcpy(void* dst, const void* src, unsigned long len) {
    unsigned char* d = (unsigned char*)dst;
    const unsigned char* s = (const unsigned char*)src;
    for (unsigned long i = 0; i < len; i++) d[i] = s[i];
    return dst;
}

__attribute__((noreturn))
void __stack_chk_fail(void) {
    SPEKI_LOG(LOG_CRIT, "stack smash");
    raw_exit_group(0xFE);
    __builtin_unreachable();
}