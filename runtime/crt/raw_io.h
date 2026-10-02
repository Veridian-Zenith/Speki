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
// These are HEX, and that is deliberate. Two other spellings look reasonable
// and both break the build somewhere it matters:
//
//   0100, 0200, ...   old-style octal. C26 deprecates it
//                     (-Wdeprecated-octal-literals), which under -Werror with
//                     -std=c2y is a hard error.
//   0o100, 0o2000000  the C23 spelling, so also deprecated-clean. But the
//                     0o prefix only became a supported spelling in clang 21,
//                     and Ubuntu 24.04 -- what CI runs on -- ships clang 18,
//                     which rejects it outright:
//                         error: invalid suffix 'o2000000' on integer constant
//
// Hex sidesteps both: no dialect dependence, no compiler-version dependence,
// and the digits line up with the kernel's own UAPI headers.
#define O_RDONLY_RAW   0
#define O_WRONLY_RAW   1
#define O_RDWR_RAW     2
#define O_CREAT_RAW    0x40
#define O_EXCL_RAW     0x80
#define O_TRUNC_RAW    0x200
#define O_APPEND_RAW   0x400
#define O_NONBLOCK_RAW 0x800
#define O_CLOEXEC_RAW  0x80000
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