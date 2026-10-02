# Speki

A custom AI model runtime, built from the bottom up: freestanding x86-64
inference with no libc, no libgcc, and no compiler-rt. Kernels are
hand-written AVX2/FMA/F16C.

**Status: WIP, deliberately starting small.** The substrate and the math
kernels are done and tested. The model is not: there is no tokenizer, no
weight loading, and no forward pass yet. The intent is a small language model
first — the runtime has to be correct on a few million parameters before it is
asked to be fast on billions. See [Roadmap](#roadmap) and
[Current state](#current-state).

The ordering is the point. Everything below the forward pass has to be exactly
right, because a kernel that is 1e-6 off does not crash — it quietly degrades
every weight it touches, and there is no debugger for that.

## Design constraints

Three rules shape everything here:

1. **CPU-only is the primary target, and it must work everywhere.** Not
   "works on the machine it was built on." The binary is fully static and
   freestanding; it runs on any x86-64 CPU matching the chosen `-march`.
2. **No bloat.** No libm. No coefficient tables. No lookup tables. No
   minimax/Remez fitting. No external dependencies of any kind. If the FPU can
   do it directly, we use it rather than approximating it in software.

Universal means universal. Anything that only works when a vendor library
happens to be installed is a portability bug, not a feature.

3. **Small first.** A model that fits in memory and runs in seconds can be
   debugged end to end. Optimising for a size you cannot yet run is how
   projects end up with fast kernels and an unrunnable model.

## Quick start

```sh
cmake -B build/cmake --preset dev
cmake --build build/cmake
./build/speki
```

Requires clang, lld, cmake >= 3.20, ninja. Nothing else.

`build.sh` still exists as a fallback if cmake is unavailable, but it is not
the primary path and its flags are not kept in sync automatically.

## Roadmap

Rough order of work. Each layer is only worth building once the one below it
is trustworthy.

- [x] Freestanding substrate: crt0, syscalls, arena, logging
- [x] Tensor types (1D/2D/3D, fp32/fp16/i8)
- [x] Math kernels: dot, gemm, silu, softmax, rmsnorm, exp
- [ ] Remaining transformer pieces: `layernorm`, `gelu`, `rope`, attention
- [ ] Layer primitives: embedding lookup, linear, MLP block, residual
- [ ] Weight loading (a simple format first, then GGUF/safetensors)
- [ ] Tokenizer (BPE, byte-level)
- [ ] Forward pass + logits
- [ ] Generation loop: sampling, KV cache
- [ ] A small model, end to end
- [ ] Quantisation (`tools/quantize`, currently an empty directory)
- [ ] Optional BLAS backend behind a seam in `raw_blas.h`

`kernels.h` already advertises `layernorm_f32`, `gelu_f32`, `rope_f32` and
`attention_f16_f32` in its header comment. They do not exist yet.

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

Everything here is accurate as of the last commit; the first group is a list
of what still has to be built.

**Not built yet:**

- **No model.** No tokenizer, no weight loading, no forward pass, no
  generation loop. `tools/quantize/` is an empty directory.
- **No BLAS seam.** `runtime/crt/raw_blas.h` is a comment-only stub, so there
  is no backend abstraction yet. When it arrives, the contract should be
  BLAS-shaped rather than MKL-shaped: MKL is a performance choice on machines
  that have it, not a portability requirement. Note that oneAPI BLAS is still
  CPU — an iGPU would need SYCL behind the same interface, not a rewrite.
- **Missing kernels.** `layernorm_f32`, `gelu_f32`, `rope_f32`,
  `attention_f16_f32` and `raw_blas.h` are referenced in comments but do not
  exist. `rope` is unusable without the exp in `mathf.h`, which now works.

**Deliberate limitations:**

- Quantisation is not implemented, so weights are effectively fp32. The FP16
  paths in the kernels are exercised by tests but no file format produces
  them yet.
- `corpus/` is empty, so the kernels are validated on synthetic inputs rather
  than on real activation distributions.
- Only x86-64. Nothing in the design is x86-specific except the ISA choices,
  but nothing has been ported either.

**Fixed recently** (all were live bugs, kept here because they explain why the
test suites look the way they do):

- `softmax_f32` called `hsum8` where it needed `hmax8`, computing the *sum* of
  the row as the maximum (27.6 where the true max was 4.5). Every output then
  saturated and the row went uniform.
- `silu_f32` and `softmax_f32` evaluated their polynomial as `y^k*poly + c_k`
  instead of Horner's `poly*y + c_k` — a different expression, returning
  negative values where the result is positive.
- `f16_to_f32` cast the bit pattern to `__fp16` before calling `_cvtsh_ss`,
  which is a *value* conversion, not a bit reinterpretation. Every fp16→fp32
  promotion was wrong (`0x4248` came back as 3.125 instead of 3.140625), which
  silently corrupted `gemm_f16_f32` inputs.
- `crt0.c` exited 0 unconditionally, so no kernel test had ever failed a
  build.

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
cmake -B build/cmake --preset dev       # native CPU, -O2, -Werror, symbols
cmake -B build/cmake --preset release   # x86-64-v2, -O3, stripped  (ships)
cmake -B build/cmake --preset portable  # x86-64-v2 baseline
cmake -B build/cmake --preset asan      # sanitizers, host tests only
```

`SPEKI_ARCH` defaults to `native`, which is right for local work and wrong for
anything built for hardware you don't control — CI and `release` therefore set
it explicitly. See [CMakePresets.md](CMakePresets.md).

Any value can be overridden from the environment, which takes precedence over
the preset:

```sh
SPEKI_ARCH=native SPEKI_OPT=O1 cmake -B build/cmake --preset dev
```

| Variable | Default | Meaning |
|---|---|---|
| `SPEKI_CSTD` | `c2x` | C dialect. clang 23 has no `-std=c26`; see below |
| `SPEKI_ARCH` | `native` | `-march` / `-mtune` target. CI and `release` set this explicitly |
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
> silicon. `-march=native` here resolves to exactly what this CPU supports, so
> it is safe — but naming an arch that *does* have AVX-512 (`skylake-avx512`,
> `sapphirerapids`, …) and running the result on hardware without it will
> SIGILL, and the build cannot detect that. CI learned this the hard way: it
> runs on AMD EPYC (Zen 3), not Alder Lake.

## Testing

Two suites answering different questions:

```sh
./build/speki                                    # freestanding: no libc at all
cmake --build build/cmake --target speki_kernels_host
./build/speki_kernels_host                       # accuracy vs fp128 reference
ctest --preset dev                               # the same suite via ctest
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