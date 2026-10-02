// SPDX-License-Identifier: OSL-3.0
// types.h — Speki fundamental types.
//
// All our headers expect these typedefs to exist. They're declared in
// raw_syscalls.h for the C runtime layer; the ML/runtime layer adds the
// floating-point types so we don't need to include any libc math header.

#ifndef SPEKI_TYPES_H
#define SPEKI_TYPES_H

#include "../crt/raw_syscalls.h"

// Floating-point primitives matching IEEE-754 binary layout.
// fp16 — half-precision. We store it as a u16 bit-pattern; conversion to
//        fp32 happens via F16C intrinsics in kernels.h.
typedef u16 fp16;

// fp32, fp64 — direct aliases for float/double. We use our own names so
// the rest of the codebase doesn't depend on <float.h> or math headers.
typedef float  fp32;
typedef double fp64;

#endif // SPEKI_TYPES_H