# Toolchain

What speki needs to build, what each piece is for, and which knobs exist.

## Requirements

| Tool | Why | Notes |
|---|---|---|
| clang >= 23 | the compiler | `-std=c2y` (C26) and `0o` literals need clang 21+ |
| lld | the linker | faster, and different default-script behaviour from GNU ld |
| cmake >= 3.20 | build system | 3.21+ if you want `--list-presets` |
| ninja | build driver | optional; make works |

Arch Linux / CachyOS package names: `clang`, `lld`, `cmake`, `ninja`.

```sh
sudo pacman -S clang lld cmake ninja
```

Debian / Ubuntu: `clang` (needs 23+; see below), `lld`, `cmake`, `ninja-build`.

**clang 23 specifically.** The build REQUIRES `-std=c2y` (C26) and fails with an
actionable message otherwise rather than downgrading. `x86-64-v2` style
`0o` literals need clang 21+. Most distributions ship clang 14-18, so on
Debian-family systems install a current clang first:

```sh
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 23
```

CI does the same via `.github/actions/setup-llvm`, which also verifies the
apt.llvm.org signing key against a pinned fingerprint.

## Configuring

```sh
cmake -B build/cmake --preset dev      # configure
cmake --build build/cmake              # build  -> build/speki
./build/speki                          # run the freestanding tests
```

Output lands in `build/speki`, not `build/cmake/speki`, so it sits beside the
optional `build.sh` output in `build/legacy/` without either overwriting the
other.

## Knobs

Every one is a CMake cache variable and can come from the environment, which
takes precedence over a preset, which takes precedence over the built-in
default.

| Variable | Default | Meaning |
|---|---|---|
| `SPEKI_CSTD` | `c2y` | C dialect. Detected; override to pin. |
| `SPEKI_ARCH` | `native` | `-march` / `-mtune` target. |
| `SPEKI_OPT` | `O3` | Optimization level. |
| `SPEKI_WERROR` | `OFF` | Warnings as errors. The `dev` preset turns this on. |
| `SPEKI_STRIP` | `ON` | Strip the binary. |
| `SPEKI_HOST_TESTS` | `ON` | Build the libm-free accuracy suite. |
| `SPEKI_LD` | `lld` | Linker driver. |
| `SPEKI_EXTRA_CFLAGS` | — | Appended to every compile. |
| `SPEKI_EXTRA_LDFLAGS` | — | Appended to every link. |

```sh
SPEKI_ARCH=native SPEKI_OPT=O1 cmake -B build/cmake --preset dev
cmake -B build/cmake --preset dev -DSPEKI_ARCH=haswell
```

The configure step prints the resolved configuration, so a build directory
never has to be reverse-engineered. See `cmake/SpekiFlags.cmake` for the flag
policy and `CMakePresets.md` for preset detail.

## Architecture selection

This is the part with sharp edges, so it gets its own section.

**AVX2 is required. There is no SSE fallback.** Every kernel is hand-written
with AVX2/FMA/F16C intrinsics, so targeting an older `-march` does not make the
binary slower - it fails to compile:

```
error: always_inline function '_mm256_fmadd_ps' requires target feature 'fma'
error: AVX vector return of type '__m256' without 'avx' enabled changes the ABI
```

So the floor is `haswell` (AVX2 + FMA + F16C, and nothing newer). That still
buys a lot of portability: no AVX-512, no VNNI, no GFNI/VAES/VPCLMULQDQ, no
Alder Lake tuning.

| Context | Arch | Why |
|---|---|---|
| Local default | `native` | fastest thing this machine can run |
| CI | `haswell` | CI must pass on hardware we do not control |
| `release` preset | `haswell` | the shipping binary |

**Do not rely on `native` for anything you ship.** `native` on a build machine
means "whatever CPU this machine happens to be", and that changes without
notice. This is not hypothetical: CI ran on an AMD EPYC 7763 (Zen 3) while
development was on Intel Alder Lake, and a binary built for `alderlake` died
there with SIGILL partway through the kernel tests.

To build something genuinely portable, write SSE paths for the kernels. That is
real work and is on the roadmap, not a flag change.

### The `-m` flags are load-bearing

`speki_apply_flags` passes `-mavx2 -mfma -mf16c` explicitly. They look
redundant under any `-march` that implies them. They are not, and removing them
was tried:

```sh
# with only -march=nehalem
error: always_inline function '_mm256_fmadd_ps' requires target feature 'fma'
```

An explicit `-m<feature>` also silently RE-ENABLES that feature regardless of
`-march`, so `SPEKI_ARCH` is not the only thing selecting the ISA. Keep both in
sync when you touch either.

### Not `x86-64-v2`

The psABI feature-level spellings read better but do not exist in every clang
build. The apt.llvm.org one lists CPUs straight from `znver6` to plain
`x86-64` with no v2/v3/v4 levels at all. `speki_validate_arch()` catches this
at configure time:

```
speki: this compiler does not accept -march=x86-64-v2.
  clang -march=help
```

## Testing

```sh
./build/speki                              # 49 freestanding tests
cmake --build build/cmake --target speki_kernels_host
./build/speki_kernels_host                 # 15 accuracy checks
ctest --preset dev
clang-tidy -p build/cmake runtime/crt/*.c runtime/runtime/*.c
```

The two suites answer different questions and neither subsumes the other. The
freestanding binary has no libm, so it checks structural properties; the host
suite checks numerical accuracy against an independent `__float128` reference.
A kernel has to satisfy both. See
[CONTRIBUTING.md](../CONTRIBUTING.md#the-rule-that-matters-most) for why.

The host suite links no C library. It needs libgcc for the soft-fp128 helpers,
which are compiler support routines and never reach the freestanding binary.

## Editing `.clang-tidy`

The check list is deliberately narrow - see the comments in the file for why.
One operational rule that has bitten twice already:

**Only checks listed under `Checks:` may appear as suppression keys.** clang-tidy
refuses to run at all if a suppression names a check that is not enabled, with
a bare `error: unknown key '<name>'` and then `Error: no checks enabled`. If you
want to silence something, add it to `Checks:` first, then suppress it.

## Legacy `build.sh`

Kept as a fallback if cmake is unavailable. Not the primary path, and its flags
are NOT kept in sync automatically - if you change compiler flags, change
`cmake/SpekiFlags.cmake` and mirror them here by hand. It writes to
`build/legacy/`.