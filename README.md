# Speki

A custom AI model runtime: freestanding x86-64 inference with no libc, no
libgcc, and no compiler-rt. Kernels are hand-written AVX2/FMA/F16C.

**Status: WIP.** What exists today is a tested kernel and runtime substrate.
There is no model, no forward pass, and no tokenizer yet. See
[Current state](#current-state) for exactly what works and what does not.

## Design constraints

Two rules shape everything here:

1. **CPU-only is the primary target, and it must work everywhere.** Not
   "works on the machine it was built on." The binary is fully static and
   freestanding; it runs on any x86-64 CPU matching the chosen `-march`.
2. **No bloat.** No libm. No coefficient tables. No lookup tables. No
   minimax/Remez fitting. No external dependencies of any kind. If the FPU can
   do it directly, we use it rather than approximating it in software.

Universal means universal. Anything that only works when a vendor library
happens to be installed is a portability bug, not a feature.

## Quick start

```sh
cmake -B build/cmake --preset dev
cmake --build build/cmake
./build/speki
```

Or with `just`, if you prefer:

```sh
just configure && just build && just test
```

Requires clang, lld, cmake >= 3.20, ninja. Nothing else.

## Current state

### Works

| Component | Status |
|---|---|
| crt0 / entry | Hand-written `_start`, correct 16-byte stack alignment |
| Syscall layer | Raw x86-64 syscalls, no libc wrappers |
| Arena allocator | Bump allocator with alignment control |
| Tensor types | 1D/2D/3D, fp32/fp16/i8, stride and byte-cap invariants |
| `dot_f32_f32`, `dot_f16_f32` | AVX2 + F16C, tested against scalar reference |
| `gemm_f16_f32` | Cache-blocked, F16C promotion, tested |
| `rmsnorm_f32` | Tested |
| `residual_add_f32` | Tested |
| `mathf.h` exp | Hand-rolled, ~1 ulp of fp32, scalar + AVX2 |

### Known-broken or missing

These are real gaps, listed rather than glossed over:

- **`silu_f32` and `softmax_f32` are wrong.** `softmax_f32` calls `hsum8`
  where it must call `hmax8`, so it computes the *sum* of the row as the
  maximum; every `exp(x - sum)` then saturates and the output goes uniform.
  Both kernels also evaluate their polynomial as `y^k * poly + c_k` instead
  of Horner's `poly * y + c_k`, which is a different expression entirely and
  returns negative values where the result should be positive.
- **`kernels_test.c` cannot fail the build.** `crt0.c` calls
  `raw_exit_group(0)` unconditionally, so the four failing kernel tests have
  never actually gated anything. The comment claiming otherwise is wrong.
- **`ref_silu` in `kernels_test.c` is itself wrong** — it clamps to `e = 1.0`
  whenever `-x >= 0`, returning `x/2` for every negative input. The test was
  comparing two broken implementations.
- **Documented-but-absent kernels.** `kernels.h`'s header comment advertises
  `gelu_f32`, `rope_f32`, `attention_f16_f32`, and `layernorm_f32`. None of
  them exist. `rope` in particular is unusable without a correct `exp`.
- **No model.** No tokenizer, no weight loading, no layer loop, no forward
  pass. `tools/quantize/` is an empty directory.
- **No BLAS seam.** `runtime/crt/raw_blas.h` is a comment-only stub. There is
  no backend abstraction yet.

## Architecture

```
crt0.c ──► _start (hand-written asm)
             │
             ▼
        speki_main ──► runs kernel + tensor test suites
             │
             ▼
   runtime/runtime/          runtime/crt/
   ├── kernels.h             ├── raw_syscalls.h   raw kernel ABI
   │   AVX2/FMA/F16C         ├── raw_alloc.h      arena allocator
   ├── mathf.h               ├── raw_log.h        logging over syscalls
   │   libm-free exp         ├── raw_io.h         stdio replacement
   ├── f16c.h                ├── raw_time.h       clock_gettime
   │   fp16 ↔ fp32           ├── raw_thread.h     minimal threading
   └── tensor.h              └── raw_blas.h       (stub — no backend yet)
       1D/2D/3D tensors
```

`docs/architecture.md` covers the freestanding constraints in detail — why
`_start` is hand-written, why there is no stack protector, and what the arena
buys us.

## Building

Everything about the toolchain is configurable. Presets cover the common
cases; every value can also come from the environment.

```sh
cmake -B build/cmake --preset dev       # -O2, -Werror, host tests, symbols
cmake -B build/cmake --preset release   # -O3, stripped
cmake -B build/cmake --preset native    # -march=native for this machine
cmake -B build/cmake --preset portable  # x86-64-v2 baseline
cmake -B build/cmake --preset asan      # sanitizers, host tests only
```

Any value can be overridden from the environment, which takes precedence over
the preset:

```sh
SPEKI_ARCH=native SPEKI_OPT=O1 cmake -B build/cmake --preset dev
```

| Variable | Default | Meaning |
|---|---|---|
| `SPEKI_CSTD` | `c2x` | C dialect. clang 23 has no `-std=c26`; see below |
| `SPEKI_ARCH` | `alderlake` | `-march` / `-mtune` target |
| `SPEKI_OPT` | `O3` | Optimization level |
| `SPEKI_WERROR` | `OFF` | Warnings as errors |
| `SPEKI_STRIP` | `ON` | Strip the binary |
| `SPEKI_HOST_TESTS` | `ON` | Build the accuracy suite |
| `SPEKI_LD` | `lld` | Linker driver |
| `SPEKI_EXTRA_CFLAGS` | — | Appended to every compile |
| `SPEKI_EXTRA_LDFLAGS` | — | Appended to every link |

The configure step prints the resolved configuration, so you never have to
guess which flags a build dir picked up.

See `CMakePresets.md` for preset details and `cmake/SpekiFlags.cmake` for the
flag policy — the comments there explain *why* each flag is present.

> **C23, not C26.** speki is pure C. `-std=c++26` works on clang 23
> (`__cplusplus == 202400`) but does not apply here; `-std=c26` is *rejected*
> by this compiler. The C dialect flag is `-std=c2x`, giving
> `__STDC_VERSION__ == 202311`, i.e. C23. We ask for the ISO dialect rather
> than `gnu2x` so a GNU extension cannot creep in unremarked.

> **The arch trap.** Consumer Alder Lake has no AVX-512, no VNNI and no AMX
> silicon, which is why the default is `alderlake`. `--preset native` is safe
> on this machine. Building with an arch that *does* have AVX-512
> (`skylake-avx512`, `sapphirerapids`, …) and running the result here will
> SIGILL. The build cannot detect that mismatch.

## Testing

Two suites answering different questions:

```sh
just test        # runs inside build/speki — proves it works with NO libc
just test-host   # checks the kernels against an independent fp128 reference
just check       # both
```

The freestanding binary cannot check its own accuracy — it has no libm to
compare against, and adding one would defeat the point. So
`tests/host/kernels_accuracy.c` compiles the same header-only kernels and
compares them against `tests/host/ref.h`: a deliberately naive `__float128`
Taylor series with its own `ln(2)`, sharing neither libc nor `mathf.h`'s
method. **That suite links no C library at all**, and its reference is
independent in both senses — not glibc's `exp`, and not the same algorithm
under test.

This is not belt-and-braces. `softmax_f32` shipped with `hsum8` where it
needed `hmax8`; its own test passed because the reference in the test file
used the same broken polynomial. A suite can only catch an error if its
reference is independent of the code under test.

## Licence

OSL-3.0. See [LICENSE](LICENSE).