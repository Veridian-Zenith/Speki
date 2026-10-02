// SPDX-License-Identifier: OSL-3.0
// raw_io.h — file I/O helpers built on raw syscalls.
//
// raw_openat / raw_close / raw_read / raw_write / raw_writev are defined in
// raw_syscalls.h. This header adds higher-level helpers used by Speki:
//   raw_pread64   — positional read (useful for sharded corpus files)
//   raw_lseek     — file offset control
//   raw_pwritev   — positional gather write
// Plus the Speki-flavored flag constants (suffixed _RAW to avoid colliding
// with any libc-style definitions that may creep in later).

#ifndef SPEKI_RAW_IO_H
#define SPEKI_RAW_IO_H

#include "raw_syscalls.h"

// Speki-flavored openat flag aliases (kernel ABI constants, not redefined).
//
// These are written with the 0o prefix rather than a leading zero. A bare
// leading-zero literal is the old octal spelling, which C26 deprecates
// (-Wdeprecated-octal-literals); under -std=c2y with -Werror that is a build
// failure, not a warning. The VALUES are unchanged: 0o100 == 0100 == 64, and
// 0o2000000 == 02000000 == O_CLOEXEC.
#define O_RDONLY_RAW   0
#define O_WRONLY_RAW   1
#define O_RDWR_RAW     2
#define O_CREAT_RAW    0o100
#define O_EXCL_RAW     0o200
#define O_TRUNC_RAW    0o1000
#define O_APPEND_RAW   0o2000
#define O_NONBLOCK_RAW 0o4000
#define O_CLOEXEC_RAW  0o2000000
#define AT_FDCWD_RAW   -100

// raw_pread64(fd, buf, len, off) → bytes read, 0 on EOF, -errno on error
static inline slong raw_pread64(i32 fd, void* buf, usize len, i64 off) {
    return SPEKI_SYSCALL4(SYS_pread64, fd, buf, len, off);
}

// raw_lseek(fd, off, whence) → new offset or -errno
// whence: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END
static inline slong raw_lseek(i32 fd, i64 off, i32 whence) {
    return SPEKI_SYSCALL3(SYS_lseek, fd, off, whence);
}

#endif // SPEKI_RAW_IO_H