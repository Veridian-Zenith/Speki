#!/usr/bin/env fish
# SPDX-License-Identifier: OSL-3.0
# build.sh — invoked by `just build`. Compiles all .c files in runtime/
# into object files under build/, then links them into a single static
# binary `speki`.
#
# Why a separate script: just's recipe parser doesn't grok fish-style
# for-loops with multi-line bodies cleanly. We sidestep the parser by
# making the recipe `build` simply exec this script.

set -e

set cxx clang
set cflags -march=alderlake -mtune=alderlake -mavx2 -mfma -mf16c -mbmi -mbmi2 -madx -mgfni -mvaes -mpclmul
set cflags $cflags -O3 -fno-plt -fno-rtti -fno-exceptions
set cflags $cflags -fno-unwind-tables -fno-asynchronous-unwind-tables -fmerge-all-constants
set cflags $cflags -ftrivial-auto-var-init=zero -fstack-clash-protection -fcf-protection=branch
set cflags $cflags -ffunction-sections -fdata-sections -ffreestanding
set cflags $cflags -Wall -Wextra -Wno-unused-parameter -std=c2x

set ldflags -static -fuse-ld=lld -Wl,--strip-all -Wl,--build-id=none
set ldflags $ldflags -Wl,-z,now -Wl,-z,relro -Wl,-z,noseparate-code
set ldflags $ldflags -Wl,--gc-sections -Wl,--exclude-libs=ALL -nostdlib -nostartfiles

# All output goes to build/legacy/ so the source root stays clean and this
# never collides with the CMake out-of-source tree in build/cmake/.
set outdir build/legacy
mkdir -p $outdir

set sources
for f in runtime/crt/*.c
    set sources $sources $f
end
if test -d runtime/runtime
    for f in runtime/runtime/*.c
        set sources $sources $f
    end
end

set objects
for s in $sources
    set o $outdir/(basename $s .c).o
    set objects $objects $o
    echo "  CC $s"
    $cxx $cflags -c $s -o $o
end

echo "  LD $outdir/speki"
$cxx $ldflags $objects -o $outdir/speki
echo "  built $outdir/speki"