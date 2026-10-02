// SPDX-License-Identifier: OSL-3.0
// raw_time.h — convenience wrappers around clock_gettime + nanosleep.
//
// raw_clock_gettime and raw_nanosleep are defined in raw_syscalls.h.
// This header adds human-friendly time helpers that aren't syscalls:
//   raw_now_ns(), raw_now_ms(), raw_sleep_ns()

#ifndef SPEKI_RAW_TIME_H
#define SPEKI_RAW_TIME_H

#include "raw_syscalls.h"

// raw_now_ns() → nanoseconds since CLOCK_MONOTONIC epoch.
// On failure returns 0. Determinism: depends on kernel, but the boot-time
// epoch is per-system, so values differ across machines. Within one
// boot, deltas are reliable.
static inline u64 raw_now_ns(void) {
    struct timespec ts;
    if (raw_clock_gettime(CLOCK_MONOTONIC, &ts) < 0) return 0;
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

// raw_now_ms() → milliseconds since CLOCK_MONOTONIC epoch.
static inline u64 raw_now_ms(void) {
    return raw_now_ns() / 1000000ull;
}

// raw_sleep_ns(ns) — sleeps for the given nanoseconds via nanosleep.
// Never spin-loops.
static inline void raw_sleep_ns(u64 ns) {
    struct timespec req;
    req.tv_sec  = (long)(ns / 1000000000ull);
    req.tv_nsec = (long)(ns % 1000000000ull);
    (void)raw_nanosleep(&req, (struct timespec*)0);
}

#endif // SPEKI_RAW_TIME_H