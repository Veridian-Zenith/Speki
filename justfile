# SPDX-License-Identifier: OSL-3.0
# justfile — task runner for speki.
#
# This file deliberately holds NO compiler flags. The build is defined once, in
# CMakeLists.txt. An earlier version of this file duplicated every -march and
# -std flag, which meant three places to edit (here, build.sh, CMakeLists.txt)
# and they had already drifted apart. If you need to change how speki compiles,
# change CMakeLists.txt.
#
# The only place a flag may still be written out is tools/quantize/, which is a
# separate Cargo workspace with its own build system.

set shell := ["fish", "-c"]

# Out-of-source build tree. Everything generated lands under build/.
cmake_dir := "build/cmake"
build_dir := "build"

# Show available recipes.
default:
    @just --list

# ─── Configure ──────────────────────────────────────────────────────────────
#
# Ninja for incremental builds; clang because speki is clang-only in practice
# (it relies on -ffreestanding plus clang's exact codegen for the crt0 stack
# alignment dance, which gcc does not reproduce).

configure:
    cmake -B {{cmake_dir}} -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ─── Build ──────────────────────────────────────────────────────────────────
#
# Requires `configure` to have run at least once.

build:
    cmake --build {{cmake_dir}}

# Rebuild from scratch, including the configure step.
rebuild: configure build

# ─── Test ───────────────────────────────────────────────────────────────────
#
# Two suites, and they answer different questions:
#
#   test        — runs inside the freestanding binary (build/speki). Proves the
#                 kernels work with NO libc at all, which is the property that
#                 actually matters for this project.
#   test-host   — links the same kernels against the host libm and measures
#                 numerical error. The freestanding binary cannot check its own
#                 accuracy; this is where that is verified.
#
# `test` is the gate. `test-host` is the one that catches a kernel that is
# self-consistently wrong.

run: build
    ./{{build_dir}}/speki

test: build
    ./{{build_dir}}/speki

test-host:
    cmake --build {{cmake_dir}} --target speki_kernels_host
    ./{{build_dir}}/speki_kernels_host

# Run both suites.
check: test test-host

# ─── Inspect ────────────────────────────────────────────────────────────────

# Confirm the binary is genuinely static and freestanding — no interpreter,
# no dynamic loader, no libc.
inspect: build
    @echo "== ELF header =="
    llvm-readelf -h {{build_dir}}/speki
    @echo ""
    @echo "== size =="
    @stat -c '%s bytes' {{build_dir}}/speki
    @echo ""
    @echo "== dynamic deps (expect none) =="
    @llvm-readelf -d {{build_dir}}/speki 2>/dev/null || echo "  none - fully static"
    @echo ""
    @echo "== undefined symbols (expect none) =="
    @llvm-nm -u {{build_dir}}/speki 2>/dev/null || echo "  none"

# ─── Clean ──────────────────────────────────────────────────────────────────
#
# Removes build/ and the legacy build.sh tree. Nothing tracked by git is
# touched; see .gitignore for what is considered generated.

clean:
    rm -rf {{build_dir}}
    @echo "removed {{build_dir}}/"

# Also drop the legacy ./build.sh objects.
distclean: clean
    rm -f crt0.o crt_extras.o
    @echo "removed legacy root objects"

# ─── Legacy build.sh ────────────────────────────────────────────────────────
#
# Kept working as a fallback if CMake is unavailable, but it is no longer the
# primary path and its flags are no longer kept in sync automatically. If you
# change compiler flags, change CMakeLists.txt and mirror them here by hand.

legacy:
    ./build.sh

# ─── Static analysis ───────────────────────────────────────────────────────
#
# clang-tidy is the cleanliness gate for contributed code. The check list and
# the reasoning behind it live in .clang-tidy — read that before adding checks,
# because most of the default set is wrong for a freestanding codebase (it
# assumes a libc and a hosted toolchain).
#
# Needs compile_commands.json, which the dev preset generates.
#
#   just tidy           all sources
#   just tidy-fix       apply the automatic fixes
#
# NOTE: these recipes run under fish (see `set shell` at the top), not bash.
# Fish loops are `for f in ...; ...; end` — bash-style `do ... done` is a
# syntax error here.

tidy:
    @echo "== clang-tidy =="
    if not command -v clang-tidy >/dev/null
        echo "  clang-tidy not found — install llvm's clang-tidy"
        exit 1
    end
    set -l failed 0
    for f in runtime/crt/*.c runtime/runtime/*.c
        clang-tidy -p {{cmake_dir}} $f
        or set failed 1
    end
    if test $failed -ne 0
        echo "  clang-tidy reported problems"
        exit 1
    end
    echo "  clean"

tidy-fix:
    for f in runtime/crt/*.c runtime/runtime/*.c
        clang-tidy -p {{cmake_dir}} --fix $f
    end

# ─── Format / lint ──────────────────────────────────────────────────────────
#
# The project uses tabs-free 4-space C, 100-col comments, and no external
# formatter config, so this is a syntax check rather than a reformat.
#
# Like `tidy`, this is fish syntax.

lint:
    @echo "== syntax check =="
    for f in runtime/crt/*.c runtime/runtime/*.c
        clang -fsyntax-only -std=c2x -march=alderlake -ffreestanding \
            -Wall -Wextra -Wno-unused-parameter $f
        or exit 1
    end
    echo "  all sources parse clean"