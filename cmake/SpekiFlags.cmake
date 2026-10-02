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

# ─── Kernels ────────────────────────────────────────────────────────────────

# C standard for the runtime.
#
# THE FLAG IS c2y, NOT c26.
# ─────────────────────────────
# C26 (__STDC_VERSION__ == 202400) has no `-std=c26` spelling. clang names
# dialects after the committee draft: `-std=c2y` selects C26, and `-std=c2x`
# selects C23 (202311). `clang -std=c26` is simply rejected as invalid.
# Verified on clang 23.1.1:
#
#   -std=c2y  ->  __STDC_VERSION__ 202400   (C26)
#   -std=c2x  ->  __STDC_VERSION__ 202311   (C23)
#   -std=c26  ->  error: invalid value
#
# So we default to the newest standard this compiler can actually select:
# c2y / C26. If a toolchain lacks it, drop to c2x with SPEKI_CSTD=c2x.
#
# We ask for the ISO dialect (c2y) rather than the GNU one (gnu2y) so a stray
# GNU extension cannot creep in unnoticed: under -std=c2y clang warns about
# anything non-standard. That matters for a project whose whole premise is
# freestanding portability.
speki_default(SPEKI_CSTD        SPEKI_CSTD        "c2y"
  "C dialect: c2y (=C26, __STDC_VERSION__ 202400), c2x (=C23). Not 'c26'.")

# C++ standard, for any future C++ target (tools/quantize is Rust today, and
# speki itself is pure C, so nothing links this yet — it is here so that
# adding a C++ component does not require touching the flag policy).
#
# Unlike C, C++ DOES have a literal spelling: -std=c++26 is accepted by clang
# 23 and yields __cplusplus == 202400. Verified.
speki_default(SPEKI_CXXSTD      SPEKI_CXXSTD      "c++26"
  "C++ dialect for any C++ target: c++26, c++23, c++20. Unused while pure C.")

speki_default(SPEKI_ARCH        SPEKI_ARCH        "alderlake"
  "Value for -march and -mtune. 'alderlake' matches this project's baseline CPU.")

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
    # SPEKI_ARCH drives both -march and -mtune. The explicit -m flags below
    # are redundant under -march=alderlake but keep the intent readable and
    # survive someone overriding SPEKI_ARCH with something narrower.
    -march=${SPEKI_ARCH}
    -mtune=${SPEKI_ARCH}
    -mavx2 -mfma -mf16c
    -mbmi -mbmi2 -madx
    -mgfni -mvaes -mpclmul

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