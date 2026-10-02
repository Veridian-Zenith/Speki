// SPDX-License-Identifier: OSL-3.0
// raw_log.h — Speki structured logger.
//
// Speki's only log sink is stderr (fd=2) via raw_write(). No libc fprintfs,
// no syslog, no journald. Each log line is one or two writes.
//
// Format: <timestamp_ns> <level> <message>\n
//
// Levels:
//   "EMRG" "ALRT" "CRIT" "ERR " "WARN" "NOTE" "INFO" "DEBG"
//
// We deliberately avoid raw_writev and large stack-allocated iovec arrays —
// at high inlining pressure the compiler may place an iovec adjacent to
// the stack-protector canary slot and trigger a spurious __stack_chk_fail.
// A simple two-call write pattern is both fast and foolproof.

#ifndef SPEKI_RAW_LOG_H
#define SPEKI_RAW_LOG_H

#include "raw_syscalls.h"
#include "raw_time.h"
#include "raw_io.h"

typedef enum LogLevel {
    LOG_EMERG = 0,
    LOG_ALERT,
    LOG_CRIT,
    LOG_ERR,
    LOG_WARN,
    LOG_NOTE,
    LOG_INFO,
    LOG_DEBUG,
} LogLevel;

// Level label table — kept in .rodata via static const.
extern const char LOG_LEVEL_LABELS[8][5];

// raw_log_stderr(level, msg, msg_len) — atomic best-effort single-line write.
//
// Format: "<ts_ns> <LEVEL> <msg>\n"
//
// Implementation: format the prefix into a 64-byte stack buffer, then two
// raw_write calls (prefix + suffix). The two writes are NOT atomic together;
// we accept that — log lines can interleave under concurrent writes, but
// that's fine for stderr.
//
// no_stack_protector: we don't link glibc, so %fs is not set by anyone and
// any canary read inside this function (clang emits one when the body has
// stack arrays > 8 bytes) would segfault. We provide our own __stack_chk_fail
// in crt_extras.c as a safety net, but disabling canaries here is cleaner.
__attribute__((noinline, no_stack_protector))
static void raw_log_stderr(LogLevel lvl, const char* msg, usize msg_len) {
    char buf[64];
    usize off = 0;

    u64 ts = raw_now_ns();

    // Format ts as decimal ASCII into buf.
    if (ts == 0) {
        buf[off++] = '0';
    } else {
        char tmp[24];
        usize j = 0;
        u64 v = ts;
        while (v > 0 && j < sizeof(tmp)) {
            tmp[j++] = (char)('0' + (v % 10));
            v /= 10;
        }
        while (j > 0 && off < sizeof(buf)) buf[off++] = tmp[--j];
    }
    if (off < sizeof(buf)) buf[off++] = ' ';

    // Append 4-char level label.
    const char* lbl = LOG_LEVEL_LABELS[lvl];
    for (usize k = 0; k < 4 && off < sizeof(buf); k++) buf[off++] = lbl[k];

    if (off < sizeof(buf)) buf[off++] = ' ';

    // Flush prefix.
    (void)raw_write(2, buf, off);
    // Flush message body + newline.
    (void)raw_write(2, msg, msg_len);
    (void)raw_write(2, "\n", 1);
}

// Convenience macros.
#define SPEKI_LOG(level, msg) \
    raw_log_stderr((level), (msg), sizeof(msg) - 1)

#define SPEKI_LOG_BUF(level, buf, len) \
    raw_log_stderr((level), (buf), (len))

#endif // SPEKI_RAW_LOG_H