// SPDX-License-Identifier: OSL-3.0
// raw_syscalls.h — bare Linux syscall wrappers for x86_64.
//
// We depend on one thing: the kernel. No libc, no libpthread, no liburing.
// Each syscall is one inline asm block. We use the `syscall` instruction
// directly — no vDSO, no glibc shims.
//
// Syscall ABI (x86_64, Linux):
//   rax = syscall number
//   rdi = arg1
//   rsi = arg2
//   rdx = arg3
//   r10 = arg4 (note: NOT rcx — clobbered by syscall instruction)
//   r8  = arg5
//   r9  = arg6
//
// Return:
//   rax >= 0  → result
//   rax < 0 (high bit set) → -errno
//
// We do not support 32-bit mode. Speki is amd64-only.

#ifndef SPEKI_RAW_SYSCALLS_H
#define SPEKI_RAW_SYSCALLS_H

#include <stdint.h>

// Primitive types matching the kernel ABI exactly.
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint64_t usize;   // size_t-sized for our purposes
typedef long     slong;   // return-type-sized for syscalls

// Linux syscall numbers (x86_64). We only add what we use.
// Full reference: /usr/include/asm-generic/unistd.h is for generic arch;
// x86_64 has its own. Source of truth: arch/x86/entry/syscalls/syscall_64.tbl
// in the kernel source. Stable since ~3.10.
#define SYS_read             0
#define SYS_write            1
#define SYS_open             2
#define SYS_close            3
#define SYS_stat             4
#define SYS_fstat            5
#define SYS_lseek            8
#define SYS_mmap             9
#define SYS_mprotect        10
#define SYS_munmap          11
#define SYS_brk             12
#define SYS_rt_sigaction    13
#define SYS_rt_sigprocmask  14
#define SYS_ioctl           16
#define SYS_pread64         17
#define SYS_pwrite64        18
#define SYS_writev          20
#define SYS_access          21
#define SYS_pipe            22
#define SYS_dup2            33
#define SYS_nanosleep       35
#define SYS_getpid          39
#define SYS_clone           56
#define SYS_fork            57
#define SYS_execve          59
#define SYS_exit            60
#define SYS_exit_group      231
#define SYS_wait4           61
#define SYS_kill            62
#define SYS_uname           63
#define SYS_futex           202
#define SYS_getrandom       318
#define SYS_clock_gettime   228
#define SYS_madvise         28
#define SYS_mlock           149
#define SYS_munlock         151
#define SYS_set_robust_list 273
#define SYS_rseq            334
#define SYS_openat          257
#define SYS_prctl           157
#define SYS_io_uring_setup  425
#define SYS_io_uring_enter  426
#define SYS_io_uring_register 427

// The `syscall` instruction clobbers rcx and r11. We tell the compiler.
#define SPEKI_SYSCALL0(num)                                            \
    ({                                                                  \
        long _r;                                                        \
        __asm__ volatile (                                              \
            "syscall"                                                   \
            : "=a"(_r)                                                  \
            : "a"((long)(num))                                          \
            : "rcx", "r11", "memory"                                    \
        );                                                              \
        _r;                                                             \
    })

#define SPEKI_SYSCALL1(num, a1)                                        \
    ({                                                                  \
        long _r;                                                        \
        __asm__ volatile (                                              \
            "syscall"                                                   \
            : "=a"(_r)                                                  \
            : "a"((long)(num)), "D"((long)(a1))                         \
            : "rcx", "r11", "memory"                                    \
        );                                                              \
        _r;                                                             \
    })

#define SPEKI_SYSCALL2(num, a1, a2)                                     \
    ({                                                                  \
        long _r;                                                        \
        __asm__ volatile (                                              \
            "syscall"                                                   \
            : "=a"(_r)                                                  \
            : "a"((long)(num)), "D"((long)(a1)), "S"((long)(a2))        \
            : "rcx", "r11", "memory"                                    \
        );                                                              \
        _r;                                                             \
    })

#define SPEKI_SYSCALL3(num, a1, a2, a3)                                 \
    ({                                                                  \
        long _r;                                                        \
        __asm__ volatile (                                              \
            "syscall"                                                   \
            : "=a"(_r)                                                  \
            : "a"((long)(num)), "D"((long)(a1)), "S"((long)(a2)),       \
              "d"((long)(a3))                                           \
            : "rcx", "r11", "memory"                                    \
        );                                                              \
        _r;                                                             \
    })

#define SPEKI_SYSCALL4(num, a1, a2, a3, a4)                             \
    ({                                                                  \
        long _r;                                                        \
        register long _r10 __asm__("r10") = (long)(a4);                 \
        __asm__ volatile (                                              \
            "syscall"                                                   \
            : "=a"(_r)                                                  \
            : "a"((long)(num)), "D"((long)(a1)), "S"((long)(a2)),       \
              "d"((long)(a3)), "r"(_r10)                                \
            : "rcx", "r11", "memory"                                    \
        );                                                              \
        _r;                                                             \
    })

#define SPEKI_SYSCALL6(num, a1, a2, a3, a4, a5, a6)                     \
    ({                                                                  \
        long _r;                                                        \
        register long _r10 __asm__("r10") = (long)(a4);                 \
        register long _r8  __asm__("r8")  = (long)(a5);                 \
        register long _r9  __asm__("r9")  = (long)(a6);                 \
        __asm__ volatile (                                              \
            "syscall"                                                   \
            : "=a"(_r)                                                  \
            : "a"((long)(num)), "D"((long)(a1)), "S"((long)(a2)),       \
              "d"((long)(a3)), "r"(_r10), "r"(_r8), "r"(_r9)            \
            : "rcx", "r11", "memory"                                    \
        );                                                              \
        _r;                                                             \
    })

// ─── Public wrappers ────────────────────────────────────────────────────────

// raw_write(fd, buf, len) → bytes written, or -errno
static inline slong raw_write(i32 fd, const void* buf, usize len) {
    return SPEKI_SYSCALL3(SYS_write, fd, buf, len);
}

// raw_exit_group(code) → never returns
__attribute__((noreturn))
static inline void raw_exit_group(i32 code) {
    SPEKI_SYSCALL1(SYS_exit_group, code);
    __builtin_unreachable();
}

// raw_exit(code) — exits only the current thread (rarely used; prefer exit_group)
__attribute__((noreturn))
static inline void raw_exit(i32 code) {
    SPEKI_SYSCALL1(SYS_exit, code);
    __builtin_unreachable();
}

// raw_read(fd, buf, len) → bytes read, 0 on EOF, -errno on error
static inline slong raw_read(i32 fd, void* buf, usize len) {
    return SPEKI_SYSCALL3(SYS_read, fd, buf, len);
}

// raw_openat(dirfd, path, flags, mode) → fd or -errno
static inline i32 raw_openat(i32 dirfd, const char* path, i32 flags, u32 mode) {
    return (i32)SPEKI_SYSCALL4(SYS_openat, dirfd, path, flags, mode);
}

// 0o prefix, not bare leading-zero: C26 deprecates the old octal spelling
// (-Wdeprecated-octal-literals). Values unchanged.
#define O_RDONLY  0
#define O_WRONLY  1
#define O_RDWR    2
#define O_CREAT   0o100
#define O_EXCL    0o200
#define O_TRUNC   0o1000
#define O_APPEND  0o2000
#define O_NONBLOCK 0o4000
#define O_CLOEXEC 0o2000000
#define AT_FDCWD  -100

static inline i32 raw_close(i32 fd) {
    return (i32)SPEKI_SYSCALL1(SYS_close, fd);
}

// raw_mmap(addr, len, prot, flags, fd, offset) → mapped address or -errno
static inline void* raw_mmap(void* addr, usize len, i32 prot, i32 flags,
                             i32 fd, i64 offset) {
    return (void*)SPEKI_SYSCALL6(SYS_mmap, addr, len, prot, flags, fd, offset);
}

static inline i32 raw_munmap(void* addr, usize len) {
    return (i32)SPEKI_SYSCALL2(SYS_munmap, addr, len);
}

#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define PROT_EXEC  0x4
#define PROT_NONE  0x0
#define MAP_PRIVATE   0x02
#define MAP_SHARED    0x01
#define MAP_ANONYMOUS 0x20
#define MAP_FIXED     0x10
#define MAP_POPULATE  0x8000
#define MAP_NONBLOCK  0x10000
#define MAP_HUGETLB   0x40000
#define MAP_STACK     0x20000

// raw_madvise(addr, len, advice) → 0 or -errno
static inline i32 raw_madvise(void* addr, usize len, i32 advice) {
    return (i32)SPEKI_SYSCALL3(SYS_madvise, addr, len, advice);
}

#define MADV_NORMAL      0
#define MADV_RANDOM      1
#define MADV_SEQUENTIAL  2
#define MADV_WILLNEED    3
#define MADV_DONTNEED    4
#define MADV_HUGEPAGE    14
#define MADV_NOHUGEPAGE  15
#define MADV_DONTFORK    16
#define MADV_DOFORK      17
#define MADV_MERGEABLE   12
#define MADV_UNMERGEABLE 13

// raw_mlock(addr, len) → 0 or -errno (locks pages in RAM, never swaps)
static inline i32 raw_mlock(void* addr, usize len) {
    return (i32)SPEKI_SYSCALL2(SYS_mlock, addr, len);
}

// The build must work BOTH freestanding (no system headers at all) and
// hosted (the host test suite links against libm and includes <math.h>,
// which transitively pulls in glibc's own timespec via <sys/types.h>).
//
// Guarding on someone else's macro does NOT work in either direction. We are
// included FIRST, so any libc guard macro (_STRUCT_TIMESPEC,
// __struct_timespec_defined, _SYS_TIMESPEC_H) is still unset when we get
// here — we define the struct, and then glibc defines its own on the way to
// <sys/types.h>, which is a redefinition error. Conversely, in a freestanding
// build no libc will ever show up to set those macros for us.
//
// So the condition is not "has timespec been defined" but "am I freestanding":
//
//   freestanding -> define it ourselves, kernel ABI layout, no libc
//   hosted       -> include <time.h> and let the C library own the type
//
// Both layouts are identical on x86-64 (two longs), so code in this header
// works unchanged either way.

#ifdef __speki_freestanding__
#ifndef _SYS_TIMESPEC_H
struct timespec {
    long tv_sec;
    long tv_nsec;
};
#endif
#else
#include <time.h>
#endif

#define CLOCK_REALTIME  0
#define CLOCK_MONOTONIC 1
#define CLOCK_BOOTTIME  7

static inline i32 raw_clock_gettime(i32 clockid, struct timespec* tp) {
    return (i32)SPEKI_SYSCALL2(SYS_clock_gettime, clockid, tp);
}

// raw_getrandom(buf, len, flags) → bytes written or -errno
static inline slong raw_getrandom(void* buf, usize len, u32 flags) {
    return SPEKI_SYSCALL3(SYS_getrandom, buf, len, flags);
}

#define GRND_NONBLOCK 0x1
#define GRND_RANDOM   0x2

// raw_writev(fd, iov, iovcnt) → bytes written or -errno (gather write)
static inline slong raw_writev(i32 fd, const void* iov, i32 iovcnt) {
    return SPEKI_SYSCALL3(SYS_writev, fd, iov, iovcnt);
}

// raw_nanosleep(req, rem) → 0 or -errno
static inline i32 raw_nanosleep(const struct timespec* req, struct timespec* rem) {
    return (i32)SPEKI_SYSCALL2(SYS_nanosleep, req, rem);
}

#endif // SPEKI_RAW_SYSCALLS_H