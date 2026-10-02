# Speki architecture

Notes on why the freestanding parts look the way they do. Most of this is
"the obvious approach does not work here, and here is what does."

## The freestanding constraint

`speki` links with `-nostdlib -nostartfiles -static`. No libc, no libgcc, no
compiler-rt. That means:

- No `memcpy`, `memset`, `memmove` — we provide our own in `crt_extras.c`.
- No `__stack_chk_fail` — also ours.
- No `libm` — hence `mathf.h`.
- No `atexit`, no destructors, no `.init_array` processing.
- No buffered stdio — `raw_io.h` writes straight to file descriptors.

The payoff is a ~110 KB static binary with no dynamic dependencies at all,
with room to grow: this currently holds the crt, the arena, the tensor layer
and seven kernels, and the model runtime itself still has to fit in there too.
The CI job `Verify the binary is genuinely freestanding` checks this on
every push: no interpreter, no loader, no shared library dependencies,
no undefined symbols.

## Why `_start` is hand-written

`crt0.c` writes `_start` in inline assembly rather than letting clang emit a
prologue, and the comment explains why:

```c
__asm__ volatile (
    "sub $8, %%rsp\n"      // mimic post-`call` state: rsp is now 8-aligned
    "jmp speki_main\n"
    ::: "memory"
);
```

The Linux kernel hands `_start` a 16-byte-aligned `rsp`. But clang's prologue
for `void _start(void)` assumes the 8-byte alignment it would see after a
`call`, because a plain C function is entered with the return address already
pushed. Trusting that assumption produces an `rsp` that is 8-mod-16 at the
point control reaches `speki_main` — so the first aligned SSE store
(`vmovdqa`, `vmovaps`) inside any inlined code faults with `#GP`.

The `sub $8` restores the post-`call` state clang expects, and `jmp` rather
than `call` avoids pushing a return address we will never use.

This is subtle, platform-specific, and exactly the kind of thing that breaks
silently on a toolchain upgrade.

**This was investigated as a suspect and cleared.** When the binary began
dying with SIGILL partway through the kernel tests, stack misalignment was the
leading theory — the binary contains ~67 `vmovaps`/`vmovapd` instructions,
several stack-relative (`vmovaps %xmm0, -0x38(%rsp)`), each of which raises
`#GP` when `rsp` is misaligned.

A standalone reproduction reported the `rsp` alignment `speki_main` actually
observes, for both this sequence and an `and $-16` + `push $0` realigning
variant. Both produce `rsp mod 16 == 0`. The alignment was never the problem,
and the realigning variant was reverted rather than left in on a false premise.

The actual cause was a `-march` mismatch: CI runs on AMD EPYC (Zen 3) while
development is on Intel Alder Lake. Recorded here so nobody re-derives the
alignment theory. See [toolchain.md](toolchain.md).

## Why `-fno-stack-protector`

Every kernel is marked `no_stack_protector`. The canary is read from `%fs:0x28`,
which requires a `fs` base register — something the kernel sets up via
`arch_prctl(ARCH_SET_FS)` or a `PT_GNU_STACK`-adjacent mechanism we do not
perform. Without it, stack protector code reads garbage and aborts.

Hence the build flags: `-fcf-protection=branch` and *not* `=full`. We do not
enable CET, so emitting shadow-stack landing pads would be theatre.

## The arena allocator

`raw_alloc.h` provides a bump allocator over a single `mmap`. There is no
`free`. For an inference runtime where almost all allocation happens once at
load time (weights) or once per forward pass (activations, in a fixed
scratch arena), that is the right trade: no fragmentation, no allocator
metadata, no locking.

`crt0.c` creates a 64 KiB scratch arena for the test suites and destroys it
before the tests run.

## Why `exp` is computed in fp64

`mathf.h` implements exp without libm and without any coefficient tables. The
interesting decision is that the range reduction and the polynomial run in
**fp64**, with a single rounding down to fp32 at the end.

The textbook alternative is Cody-Waite reduction in fp32:

```
k = rint(x * log2(e))
r = x - k*ln2
result = poly(r) * 2^k
```

In fp32 this is limited to roughly 1e-5 relative error, because ln2 is not
representable in 24 bits: the `k*ln2` product carries a rounding error that
grows with `|k|`. The usual fix is to split ln2 into a high and low part and
subtract in two steps — more constants, more code, and still not exact.

Doing it in fp64 sidesteps the problem. `k*ln2` is exact in fp64 for every
`|k|` the clamped input range allows, so a single subtraction leaves a clean
argument. The polynomial is then plain Horner to degree 7 in fp64, and the
result rounds once to fp32.

AVX2 provides 4-wide fp64 natively, so the "slow" path is still 4 lanes per
instruction. The approach is both shorter and more accurate than the fp32
version it replaces — which is the whole reason `mathf.h` is small.

`2^k` is constructed by writing `k + 1023` into the exponent field rather than
calling `scalbn()`. There is no libm to call.

The 8-wide fp32 entry point runs two 4-wide fp64 passes and splices the
results. It matches the scalar path lane for lane; `tests/host/` asserts
bit-exact agreement so the two can never silently diverge.

## Numerical correctness is tested against libm

The freestanding binary cannot verify its own numerics — there is no libm to
compare against, and adding one would defeat the point. So
`tests/host/kernels_accuracy.c` compiles the *same header-only kernels* on the
host, where `exp()` is available as ground truth, and asserts relative error
against documented tolerances.

This is not belt-and-braces. `softmax_f32` shipped with `hsum8` where it
needed `hmax8` — it computed the sum of the row instead of the maximum, every
`exp(x - sum)` saturated at the clamp, and all outputs collapsed to a uniform
distribution. Its test *passed*, because the reference implementation in the
test file used the same broken polynomial. The kernel and its reference were
wrong in the same direction, so they agreed.

A test suite can only catch an error if its reference is independent of the
code under test. That is the entire reason the host suite exists.

## Build layout

```
build/
├── cmake/       CMake + Ninja tree
└── legacy/      old build.sh output
```

The binary lands in `build/` so both paths are easy to find without
colliding. `justfile` is a thin task runner over CMake and deliberately holds
no compiler flags — those live in `cmake/SpekiFlags.cmake`, once.

## Current ISA assumptions

**AVX2 is required, and there is no SSE fallback.** Every kernel is hand-written
with AVX2/FMA/F16C intrinsics, so an older `-march` does not make the binary
slower — it fails to compile:

```
error: always_inline function '_mm256_fmadd_ps' requires target feature 'fma'
error: AVX vector return of type '__m256' without 'avx' enabled changes the ABI
```

So `haswell` (AVX2 + FMA + F16C, nothing newer) is the floor. That is what CI
and the `release` preset target. It still buys real portability: **no**
AVX-512, VNNI, or AMX, and no GFNI/VAES/VPCLMULQDQ or Alder Lake tuning.

Two consequences worth stating plainly:

- Local development defaults to `-march=native`, the fastest thing this machine
  can run.
- Anything built for hardware you do not control must set `SPEKI_ARCH`
  explicitly. `native` on a build machine means "whatever CPU the runner
  happens to be". CI ran on an AMD EPYC 7763 (Zen 3) while development was on
  Intel Alder Lake, and an `alderlake` binary died there with SIGILL
  mid-test.

Getting below `haswell` means writing SSE paths for every kernel. That is real
work on the roadmap, not a flag change.

Note that `-mavx2 -mfma -mf16c` are passed explicitly and are **load-bearing**:
they state a real requirement, and an explicit `-m<feature>` also silently
re-enables that feature regardless of `-march`. See
[toolchain.md](toolchain.md#the-m-flags-are-load-bearing).

## The C standard: C26

speki is pure C, so the C dialect matters and the C++ one does not.

C26 has no `-std=c26` spelling. clang names dialects after the committee draft:
`-std=c2y` selects C26 and `-std=c2x` selects C23.

```
-std=c2y  ->  __STDC_VERSION__ 202400   (C26)
-std=c2x  ->  __STDC_VERSION__ 202311   (C23)
-std=c26  ->  error: invalid value
```

The build **requires** `c2y` and fails with an actionable message if the
compiler cannot provide it, rather than silently downgrading. That was a
deliberate reversal: the earlier version probed `c2y → c2x → c17` and took the
first that compiled, which meant a build could pass locally and quietly produce
something different on an older compiler. Old compilers also carry known CVEs,
so a pinned old toolchain is a supply-chain liability, not just an
inconvenience. `SPEKI_CSTD` can still pin a dialect explicitly, but a
downgrade has to be asked for rather than fallen into.

We request the ISO dialect rather than the GNU one so a GNU extension cannot
slip in unremarked. For a project whose premise is freestanding portability,
that warning is worth having.

C26 also deprecates bare leading-zero octal, and the replacement `0o` prefix
only became a supported spelling in clang 21 — so the open() flags in
`raw_io.h` are hex, which is the one spelling with no dialect or
compiler-version dependency.

## Hosted vs freestanding: the `__speki_freestanding__` macro

`raw_syscalls.h` must compile in both worlds:

- **Freestanding** (`speki` itself) — no system headers at all, so it defines
  its own `struct timespec` with the kernel ABI layout.
- **Hosted** (`tests/host`) — glibc is present, and `<math.h>` transitively
  pulls in glibc's own `timespec`.

Guarding on someone else's macro does not work, because we are included
*first*: any libc guard (`_STRUCT_TIMESPEC`, `__struct_timespec_defined`,
`_SYS_TIMESPEC_H`) is still unset when `raw_syscalls.h` runs, so we define the
struct and glibc then defines its own — a redefinition error. The condition
has to be "am I freestanding", not "has timespec been defined":

```c
#ifdef __speki_freestanding__
struct timespec { long tv_sec; long tv_nsec; };
#else
#include <time.h>
#endif
```

Both layouts are two longs on x86-64, so callers are unaffected either way.