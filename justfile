# SPDX-License-Identifier: OSL-3.0
# build.just — Speki build orchestration.
#
# Run: `just build` to compile the runtime into a single static ELF.
# Run: `just run`   to build and execute `./speki`.
#
# We use clang/LLVM only. lld is the linker. No libc, no GNU anything.
# Just is MIT (not GNU make).

# We use fish as the shell (per project convention). Multi-line recipes
# in fish need exactly 4-space indentation under the recipe name.
set shell := ["fish", "-c"]

cxx      := "clang"
ld       := "ld.lld"

# CFLAGS for the runtime (C/C++).
# -march=alderlake targets AVX2, FMA, F16C, BMI1/2, GFNI, ADX, VAES, VPCLMULQDQ
# -mtune=alderlake schedules for this microarch specifically
# -nostdlib: no libc, no libgcc. We use raw_syscalls.h for the kernel ABI.
# -fstack-protector-strong + -fcf-protection=full: matches your fish config.
# -ftrivial-auto-var-init=zero: matches your fish config.
# -ffreestanding: tells clang we don't have a hosted environment.
# -O3 -Otime: optimize for speed. -flto=full: link-time optimization.
# -std=c2x: C23. We use _Static_assert, _Noreturn, [[]] attributes, etc.
# -std=c++26: only for .cpp files (set per-language below).
cflags_common := "-march=alderlake -mtune=alderlake -mavx2 -mfma -mf16c -mbmi -mbmi2 -madx -mgfni -mvaes -mpclmul -O3 -fno-plt -fno-rtti -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fmerge-all-constants -ftrivial-auto-var-init=zero -fstack-clash-protection -fcf-protection=branch -ffunction-sections -fdata-sections -ffreestanding -Wall -Wextra -Wno-unused-parameter"

# Per-language flags.
cflags_c   := cflags_common + " -std=c2x"
cflags_cpp := cflags_common + " -std=c++26"

ldflags := "-static -fuse-ld=lld -Wl,--strip-all -Wl,--build-id=none -Wl,-z,now -Wl,-z,relro -Wl,-z,noseparate-code -Wl,--gc-sections -Wl,--exclude-libs=ALL -nostdlib -nostartfiles"

# Default target: just `build` does everything.
default: build

# ─── Build the runtime as a single static binary ────────────────────────────
# Sources: crt/*.c plus runtime/runtime/*.c. All other headers are
# header-only inline functions; only .c files become .o files.
#
# Build with: just build
# Build with MKL: just build blas=mkl
#
# The actual build logic lives in ./build.sh (a fish script) because just's
# recipe parser doesn't grok fish for-loops with multi-line bodies cleanly.

build *args:
    ./build.sh

# ─── Run it ─────────────────────────────────────────────────────────────────
run: build
    ./speki

# ─── Inspect the binary ─────────────────────────────────────────────────────
inspect: build
    llvm-readelf -h speki
    llvm-readelf -S speki | head -30
    @echo "---"
    @echo "size: $(stat -c%s speki) bytes"
    @echo "deps:"
    -llvm-readelf -d speki 2>/dev/null || echo "  (no dynamic deps — fully static)"

# ─── Clean ──────────────────────────────────────────────────────────────────
clean:
    rm -rf build speki

# ─── Quant tool (Rust) — separate workspace, OSL-3.0 ────────────────────────
quant:
    cd tools/quantize && cargo build --release
    @echo "  built speki-quantize → tools/quantize/target/release/speki-quantize"