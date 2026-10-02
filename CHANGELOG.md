# Changelog

Notable changes to speki. Format loosely follows Keep a Changelog; the project
is pre-1.0 and milestones are used loosely, so dates and headings are the
useful part.

## Unreleased

Nothing yet.

## 2026-10-02

### Added

- `mathf.h`: libm-free `exp`, scalar and 8-wide AVX2. Range reduction and
  polynomial run in fp64 with a single rounding to fp32, giving ~1 ulp of fp32
  with no coefficient tables and no lookup tables. Measured exact
  (`max_rel_err = 0.000e+00`) against an independent `__float128` reference.
- `tests/host/ref.h`: an independent reference for the accuracy suite. A Taylor
  series in `__float128` with its own `ln(2)`, sharing neither libc nor
  `mathf.h`'s method. The suite links no C library.
- `tests/host/kernels_accuracy.c`: 15 numerical checks against that reference.
- CMake build with presets (`dev`, `release`, `portable`, `native`, `asan`).
  Every toolchain knob is a cache variable readable from the environment.
- `cmake/SpekiFlags.cmake`: flag policy with the reasoning for each flag.
- `.clang-tidy` and a CI `tidy` gate.
- CI: build + freestanding verification, host accuracy, clang-tidy,
  release-size. `.github/dependabot.yml` for actions and submodules.
- `.github/actions/setup-llvm`: installs a current clang from apt.llvm.org with
  a pinned key fingerprint, because the runner image ships only clang 16-18.
- `README.md`, `docs/architecture.md`, `CMakePresets.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`.
- `README.md` roadmap, stating the bottom-up order and that the model comes
  last.

### Fixed

- `crt0.c` exited 0 unconditionally, so no kernel test had ever failed a
  build. Now exits 1, and `kernels_test_main` returns its failure count.
- `softmax_f32` called `hsum8` where it needed `hmax8`, computing the row sum
  as the maximum (27.6 where the true max was 4.5). Every output saturated and
  the row went uniform.
- `silu_f32` and `softmax_f32` evaluated their polynomial as `y^k*poly + c_k`
  instead of Horner's `poly*y + c_k`, a different expression which returned
  negative values where the result is positive (at `x = 3.5`: -5.4259 against a
  true 3.3974).
- `f16_to_f32` cast the bit pattern to `__fp16` before `_cvtsh_ss`, which is a
  value conversion rather than a bit reinterpretation. Every fp16 to fp32
  promotion was wrong, silently corrupting `gemm_f16_f32` inputs.
- `raw_syscalls.h` defined `struct timespec` unconditionally, colliding with
  glibc in hosted builds.
- Test bugs that hid the above: an `unsigned` underflow in the silu input
  range, a reference `exp` using fp64's exponent layout for fp128, a
  `f16c_roundtrip_pi` tolerance tighter than correct round-to-nearest, and
  `(fp32)(i - 8)` where `(i32)i - 8` was meant.

### Changed

- Kernels now use `mathf.h` rather than per-kernel polynomial approximations.
- The freestanding suite checks structural properties instead of comparing
  against a Taylor series, which diverged past `|x| ~ 2` and could not have
  validated the kernels it was meant to check.
- The C dialect is detected, not hardcoded: the build requires C26 (`-std=c2y`)
  and fails with an actionable message rather than silently downgrading.
  `0o` literals need clang 21+; the open() flags are hex for that reason.
- `SPEKI_ARCH` defaults to `native` for local work. CI and `release` set it
  explicitly to `haswell`, because `native` on a build machine means whatever
  CPU the runner happens to be. Validated at configure time by
  `speki_validate_arch()`.
- Removed `justfile`; cmake and ctest cover the same ground.

### Known limitations

- **AVX2 is required and there is no SSE fallback.** Every kernel is
  hand-written AVX2/FMA/F16C, so an older `-march` fails to compile rather
  than running slower. `haswell` is the floor today.
- `layernorm_f32`, `gelu_f32`, `rope_f32`, `attention_f16_f32` and
  `raw_blas.h` are referenced in comments but do not exist.
- No tokenizer, weight loading, forward pass or generation loop.
- `tools/quantize/` and `corpus/` are empty, so kernels are validated on
  synthetic inputs rather than real activation distributions.
- x86-64 only. Nothing in the design is x86-specific except the ISA choices,
  but nothing has been ported.