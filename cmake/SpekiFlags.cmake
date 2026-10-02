# SPDX-License-Identifier: OSL-3.0
# SpekiFlags.cmake — toolchain and compiler flags, driven by cache variables.
#
# Every knob is a cache STRING so it can be set from a preset, the command
# line, or the environment. The env is only consulted when the cache variable
# is not already set, so `-DSPEKI_ARCH=...` beats the environment, and a preset
# beats nothing. To make the environment win unconditionally, configure into a
# fresh build directory.
#
# Split out of CMakeLists.txt so the flag policy lives in one readable file
# rather than being interleaved with target definitions.

include_guard(GLOBAL)


# ─── Read an environment variable into a cache variable ─────────────────────
#
# speki_default(<cachevar> <envname> <default> [docstring])
#
# Precedence, highest first:
#
#   1. Environment        SPEKI_ARCH=native
#   2. Preset cache var   --preset dev
#   3. -D on the command line
#   4. The <default> here
#
# Note that 2 and 3 are indistinguishable once CMake has written them to the
# cache, and presets are applied before this module runs — which is why the
# environment has to win outright rather than fill in gaps. Both are ordinary
# cache entries by the time we get here.
#
# The practical consequence: if SPEKI_ARCH is exported in your shell, it
# overrides the preset. Unset it to let the preset decide:
#
#   env -u SPEKI_ARCH cmake -B build/cmake --preset release
#
# To see what a build dir was actually configured with, re-run the configure
# step (cheap, it is cached) or read build/cmake/CMakeCache.txt.

function(speki_default var env default)
  set(help "${ARGN}")
  if(DEFINED ENV{${env}} AND NOT "$ENV{${env}}" STREQUAL "")
    set(${var} "$ENV{${env}}" CACHE STRING "${help}" FORCE)
  elseif(NOT DEFINED ${var})
    set(${var} "${default}" CACHE STRING "${help}")
  endif()
endfunction()

# speki_detect_cstd
#
# Probes for the newest C dialect the compiler actually accepts, newest first,
# and sets SPEKI_CSTD. An explicit SPEKI_CSTD (or SPEKI_CSTD env var) wins and
# skips the probe entirely.
#
# Why probe instead of defaulting to c2y: c2y (C26) is recent, and older clang
# — including whatever ubuntu-latest ships — rejects it outright with
# "error: invalid value 'c2y' in '-std=c2y'". A build system that hardcodes the
# newest standard works on exactly one machine.
#
# The candidates are ISO dialects only (c2y, c2x, c17), never gnu*: speki is
# freestanding and must not silently depend on GNU extensions.

function(speki_detect_cstd)
  # We require a current toolchain rather than degrading to an older dialect.
  #
  # The earlier version of this probed c2y -> c2x -> c17 and took the first that
  # compiled. That is the wrong policy for two reasons:
  #
  #   - A silent fallback means a build that passed on clang 23 can quietly
  #     build something different on clang 18, with no signal. That is how the
  #     0o-literal discrepancy got in: local clang 23 accepted it, CI clang 18
  #     did not, and only CI found out.
  #   - Old compilers carry known CVEs. A build pinned to one is a supply-chain
  #     liability, not just a maintenance annoyance.
  #
  # So: assert the dialect, and fail here with an actionable message if the
  # compiler cannot provide it. SPEKI_CSTD may still be set explicitly, for a
  # deliberate downgrade -- but it has to be asked for, not fallen into.
  if(NOT "${SPEKI_CSTD}" STREQUAL "")
    message(STATUS "speki: C dialect pinned to ${SPEKI_CSTD}")
    return()
  endif()

  include(CheckCSourceCompiles)

  set(CMAKE_REQUIRED_FLAGS "-std=c2y")
  set(CMAKE_REQUIRED_QUIET ON)
  check_c_source_compiles("int main(void) { return 0; }" SPEKI_C2Y_AVAILABLE)
  unset(CMAKE_REQUIRED_FLAGS)

  if(SPEKI_C2Y_AVAILABLE)
    set(SPEKI_CSTD "c2y" CACHE STRING "C dialect" FORCE)
    message(STATUS "speki: using C dialect c2y (C26)")
    return()
  endif()

  message(FATAL_ERROR
    "speki: this compiler does not support -std=c2y (C26).\n"
    "CI installs a current clang from apt.llvm.org; do the same locally, or "
    "set SPEKI_CSTD=c2x to build against C23 explicitly.\n"
    "Note that clang < 21 also rejects the 0o octal prefix, so a C23 build "
    "needs the codebase kept free of C23-only spellings.")
endfunction()

# ─── Kernels ────────────────────────────────────────────────────────────────

# C standard for the runtime.
#
# THE FLAG IS c2y, NOT c26 — AND NOT EVERY COMPILER HAS IT.
# ───────────────────────────────────────────────────────
# C26 (__STDC_VERSION__ == 202400) has no `-std=c26` spelling. clang names
# dialects after the committee draft: `-std=c2y` selects C26, `-std=c2x`
# selects C23 (202311), and `-std=c26` is rejected outright.
#
# c2y only exists in newer clang. Ubuntu's default clang (as used by
# ubuntu-latest in CI) tops out at c23 and errors with
#   error: invalid value 'c2y' in '-std=c2y'
# so hardcoding the newest dialect breaks the build on any older toolchain.
# speki_detect_cstd() below probes what the compiler actually accepts and
# takes the newest one that works.
#
# SPEKI_CSTD overrides the probe entirely, if you want to pin it:
#   SPEKI_CSTD=c2x cmake -B build/cmake --preset dev
#
# We request the ISO dialect rather than the GNU one (gnu2y/gnu2x) so a stray
# GNU extension cannot creep in unnoticed: under -std=c2y clang warns about
# anything non-standard. That matters for a project whose whole premise is
# freestanding portability.
speki_default(SPEKI_CSTD_REQUESTED SPEKI_CSTD ""
  "Pin the C dialect (e.g. c2y, c2x, c17). Empty = probe for the newest the compiler supports.")

# C++ standard, for any future C++ target (tools/quantize is Rust today, and
# speki itself is pure C, so nothing links this yet — it is here so that
# adding a C++ component does not require touching the flag policy).
#
# Unlike C, C++ DOES have a literal spelling: -std=c++26 is accepted by clang
# 23 and yields __cplusplus == 202400. Verified.
speki_default(SPEKI_CXXSTD      SPEKI_CXXSTD      "c++26"
  "C++ dialect for any C++ target: c++26, c++23, c++20. Unused while pure C.")

# TARGET ARCHITECTURE
# ─────────────────────
# Default: native. This project is developed on one machine, so the fastest
# thing for a local build is what that machine can actually run -- including
# instructions newer than any named baseline.
#
# CI and any build meant to be shipped set this explicitly, via -DSPEKI_ARCH= or
# the SPEKI_ARCH env var. Never rely on the default there: "native" on a build
# machine means "whatever CPU the CI runner happens to be", and those change
# without notice.
speki_default(SPEKI_ARCH        SPEKI_ARCH        "native"
  "Value for -march/-mtune: native, x86-64-v2, x86-64-v3, alderlake, ...")

speki_default(SPEKI_OPT         SPEKI_OPT         "O3"
  "Optimization level passed to -<value>.")

speki_default(SPEKI_WERROR      SPEKI_WERROR      "OFF"
  "Treat compiler warnings as errors.")

speki_default(SPEKI_EXTRA_CFLAGS SPEKI_EXTRA_CFLAGS ""
  "Extra flags appended verbatim to every compile command.")

speki_default(SPEKI_HOST_TESTS  SPEKI_HOST_TESTS  "ON"
  "Build the host-side kernel accuracy suite (needs libm).")

# ─── Linker ─────────────────────────────────────────────────────────────────

speki_default(SPEKI_LD          SPEKI_LD          "lld"
  "Linker driver: lld, ld, gold, or empty for the compiler default.")

speki_default(SPEKI_STRIP       SPEKI_STRIP       "ON"
  "Strip the linked binary.")

speki_default(SPEKI_EXTRA_LDFLAGS SPEKI_EXTRA_LDFLAGS ""
  "Extra flags appended verbatim to every link command.")

# ─── Applied before add_executable's flags are finalized ───────────────────

# speki_apply_toolchain(<target>)
#
# Sets the C standard and anything that must be in place before the target's
# own options are assembled.

# speki_validate_arch
#
# Check that the compiler actually accepts -march=<SPEKI_ARCH>, and fail at
# CONFIGURE time with an actionable message if it does not.
#
# This is not hypothetical. The build asked for -march=x86-64-v2, which the local
# upstream clang accepts but the apt.llvm.org build does not -- its valid-CPU
# list goes straight from znver6 to plain x86-64, with no v2/v3/v4 levels, since
# those are upstream-only spellings of the psABI feature levels. The mismatch
# surfaced as four "unknown target CPU" errors partway through every CI build,
# after the toolchain had already been installed.
#
# Checking here costs one compiler invocation and turns a confusing mid-build
# failure into a single clear line at configure time.
function(speki_validate_arch)
  # A compiler that cannot parse even a trivial program for this arch will fail
  # here; that is the point.
  set(CMAKE_REQUIRED_FLAGS "-march=${SPEKI_ARCH}")
  set(CMAKE_REQUIRED_QUIET ON)
  check_c_source_compiles("int main(void){return 0;}" SPEKI_ARCH_OK_${SPEKI_ARCH})
  unset(CMAKE_REQUIRED_FLAGS)

  if(NOT SPEKI_ARCH_OK_${SPEKI_ARCH})
    message(FATAL_ERROR
      "speki: this compiler does not accept -march=${SPEKI_ARCH}.\n"
      "List the CPUs it knows with:\n"
      "  ${CMAKE_C_COMPILER} -march=help\n"
      "Common portable baselines: nehalem (SSE4.2 era, ~2008+), "
      "sandybridge, x86-64 (plain baseline).\n"
      "Note that the psABI spellings x86-64-v2/v3/v4 exist only in upstream "
      "clang, not in every distribution build.")
  endif()
  message(STATUS "speki: target CPU ${SPEKI_ARCH} accepted")
endfunction()

function(speki_apply_toolchain target)

  # Set the dialect explicitly rather than via target_compile_features, which
  # would silently pick gnu23 and hide whether the ISO or GNU dialect is in
  # effect. SPEKI_CSTD defaults to c2x (=C23).
  target_compile_options(${target} PRIVATE "-std=${SPEKI_CSTD}")

  # lld is preferred over the default GNU ld: it is faster, and it does not
  # pull in the same default-script behavior. SPEKI_LD= (empty) falls back to
  # whatever the compiler driver chooses.
  if(NOT SPEKI_LD STREQUAL "")
    target_link_options(${target} PRIVATE "-fuse-ld=${SPEKI_LD}")
  endif()
endfunction()

# speki_apply_cxx_flags(<target>)
#
# Applies the C++ dialect and the C++-relevant subset of the flags. Called by
# any C++ target; there are none today, which is why this is separate from
# speki_apply_flags (which sets -ffreestanding and C-only diagnostics).
function(speki_apply_cxx_flags target)
  target_compile_options(${target} PRIVATE "-std=${SPEKI_CXXSTD}")
  speki_apply_flags(${target})
  # C++ needs these; they were harmless in C but are not optional here.
  target_compile_options(${target} PRIVATE -fno-rtti -fno-exceptions)
endfunction()

# ─── Flag assembly ──────────────────────────────────────────────────────────

# speki_apply_flags(<target>)
#
# Compile options, then link options. The lists below are the project's actual
# policy — read the comments, they explain WHY each flag is here rather than
# restating what the flag is called.

function(speki_apply_flags target)
  target_compile_options(${target} PRIVATE
    # ── ISA selection ──
    # SPEKI_ARCH drives both -march and -mtune.
    #
    # The explicit -m flags below ARE redundant under any -march that implies
    # them, and they were briefly removed to see whether the arch setting alone
    # was enough. It is not: the kernels are hand-written AVX2/FMA/F16C, and
    # with only -march=nehalem the build fails with
    #
    #   always_inline function '_mm256_fmadd_ps' requires target feature 'fma'
    #   AVX vector return of type '__m256' without 'avx' enabled changes the ABI
    #
    # There is no SSE fallback path in this codebase, so the build genuinely
    # requires AVX2 + FMA + F16C. Setting these explicitly is honest about that:
    # it says "this project needs these three extensions", rather than relying on
    # a particular -march to imply them. A baseline without AVX2 is not
    # supported yet -- see the roadmap.
    -march=${SPEKI_ARCH}
    -mtune=${SPEKI_ARCH}
    -mavx2 -mfma -mf16c

    # ── Optimization ──
    -${SPEKI_OPT}

    # ── Freestanding / no hosted runtime ──
    # There is no libc. -ffreestanding stops clang assuming hosted semantics
    # (notably that memcpy/memset exist), and -nostdlib/-nostartfiles below do
    # the real work at link time.
    -ffreestanding
    -fno-plt

    # ── C++-only flags, harmless for C but kept for symmetry ──
    -fno-rtti
    -fno-exceptions

    # ── Unwind tables ──
    # We never unwind: there is no C++ exception machinery and no backtrace
    # consumer. Dropping the tables removes ~10% of binary size.
    -fno-unwind-tables
    -fno-asynchronous-unwind-tables

    # ── Code/data layout ──
    # One section per function/data object, so the linker's --gc-sections can
    # actually drop unreferenced code. Without these, gc-sections can only
    # work at page granularity and much dead weight survives.
    -ffunction-sections
    -fdata-sections
    -fmerge-all-constants

    # ── Hardening ──
    # Branch-only CFG: NOT =full. We do not enable CET/shadow stack, so asking
    # for =full would emit endbr64 landing pads for a feature that is off.
    -fcf-protection=branch
    # Zero-init locals rather than leaving them indeterminate — matters because
    # this runtime has no libc memset to be sure the compiler emitted one.
    -ftrivial-auto-var-init=zero
    -fstack-clash-protection

    # ── Diagnostics ──
    -Wall
    -Wextra
    # Kernels take deliberately-unused params (NULL gamma, etc.) by design.
    -Wno-unused-parameter
  )

  if(SPEKI_WERROR)
    target_compile_options(${target} PRIVATE -Werror)
  endif()

  if(NOT SPEKI_EXTRA_CFLAGS STREQUAL "")
    # Split on whitespace so a single env var can carry several flags.
    separate_arguments(SPEKI_EXTRA_CFLAGS_LIST UNIX_COMMAND "${SPEKI_EXTRA_CFLAGS}")
    target_compile_options(${target} PRIVATE ${SPEKI_EXTRA_CFLAGS_LIST})
  endif()

  target_link_options(${target} PRIVATE
    # Fully static, no startup files, no libc. This is what makes the binary
    # runnable as a plain ./speki with no loader involved.
    -static
    -nostdlib
    -nostartfiles

    # Drop unreferenced sections (see -ffunction-sections above) and exclude
    # symbols from any static archive that sneaks in.
    "LINKER:--gc-sections"
    "LINKER:--exclude-libs=ALL"

    # ── Link-time hardening ──
    # now:   full RELRO — every GOT entry resolved at load, no lazy binding.
    # relro: mark the GOT read-only after relocation.
    "LINKER:-z,now"
    "LINKER:-z,relro"

    # Compact output. noseparate-code avoids per-section padding, which is a
    # large fraction of a 40 KB binary.
    "LINKER:-z,noseparate-code"
    "LINKER:--build-id=none"
  )

  if(SPEKI_STRIP)
    target_link_options(${target} PRIVATE "LINKER:--strip-all")
  endif()

  if(NOT SPEKI_EXTRA_LDFLAGS STREQUAL "")
    separate_arguments(SPEKI_EXTRA_LDFLAGS_LIST UNIX_COMMAND "${SPEKI_EXTRA_LDFLAGS}")
    target_link_options(${target} PRIVATE ${SPEKI_EXTRA_LDFLAGS_LIST})
  endif()
endfunction()

# ─── Report ─────────────────────────────────────────────────────────────────

function(speki_print_config)
  message(STATUS "speki build configuration:")
  message(STATUS "  compiler        ${CMAKE_C_COMPILER}")
  message(STATUS "  C standard      ${SPEKI_CSTD} (C++ ${SPEKI_CXXSTD} if used)")
  message(STATUS "  arch (-march)   ${SPEKI_ARCH}")
  message(STATUS "  optimization    -${SPEKI_OPT}")
  message(STATUS "  linker          ${SPEKI_LD}")
  message(STATUS "  werror          ${SPEKI_WERROR}")
  message(STATUS "  strip           ${SPEKI_STRIP}")
  message(STATUS "  host tests      ${SPEKI_HOST_TESTS}")
  if(NOT SPEKI_EXTRA_CFLAGS STREQUAL "")
    message(STATUS "  extra cflags    ${SPEKI_EXTRA_CFLAGS}")
  endif()
  if(NOT SPEKI_EXTRA_LDFLAGS STREQUAL "")
    message(STATUS "  extra ldflags   ${SPEKI_EXTRA_LDFLAGS}")
  endif()
endfunction()