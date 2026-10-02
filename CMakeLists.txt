# SPDX-License-Identifier: OSL-3.0
# CMakeLists.txt — Speki build.
#
# Produces a single fully-static freestanding ELF: `speki`.
# No libc, no libgcc, no compiler-rt. Entry point is our own `_start`.
#
#   cmake -B build/cmake --preset dev      # see CMakePresets.json
#   cmake --build build/cmake
#   ./build/speki
#
# ─── ADJUSTING THE BUILD ────────────────────────────────────────────────────
#
# Nothing about the toolchain is baked in. Every flag below is a cache
# variable, so you can override any of them three ways, in increasing
# precedence:
#
#   1. A CMake preset            cmake -B build/cmake --preset release
#   2. A cache variable          cmake -B build/cmake -DSPEKI_ARCH=native
#   3. The environment           SPEKI_ARCH=native cmake -B build/cmake
#
# Environment wins over the command line here, because the env is read at
# configure time into the cache only when the variable is NOT already set.
# To force an env override, configure into a fresh build dir.
#
# The knobs you will actually want:
#
#   SPEKI_ARCH        -march / -mtune target.      default: alderlake
#   SPEKI_OPT         optimization level.          default: O3
#   SPEKI_CC          compiler binary.             default: clang
#   SPEKI_LD          linker driver flavor.        default: lld
#   SPEKI_EXTRA_CFLAGS     appended verbatim to every compile
#   SPEKI_EXTRA_LDFLAGS    appended verbatim to every link
#   SPEKI_WERROR      treat warnings as errors.    default: OFF
#   SPEKI_STRIP       strip the binary.            default: ON
#   SPEKI_HOST_TESTS  build the libm-comparison suite.  default: ON
#
# Example — build for whatever CPU this machine actually has, and turn
# warnings into errors:
#
#   SPEKI_ARCH=native SPEKI_WERROR=ON cmake -B build/cmake --preset dev
#
# IMPORTANT: -march=alderlake is the default because consumer Alder Lake has
# no AVX-512/VNNI/AMX silicon. Setting SPEKI_ARCH=native on such a machine is
# safe; setting it to a desktop Skylake-X (which does have AVX-512) while
# running on Alder Lake WILL produce a binary that SIGILLs at startup. The
# build cannot detect this for you.

cmake_minimum_required(VERSION 3.20)
project(speki C)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpekiFlags.cmake")

# Resolve the C dialect before anything reads SPEKI_CSTD or emits a -std flag.
# This has to run at directory scope: check_c_source_compiles() caches its
# result, and calling it from inside a function() would scope the SPEKI_CSTD
# assignment to that function.
speki_detect_cstd()
speki_validate_arch()

speki_print_config()

# ─── Sources ────────────────────────────────────────────────────────────────
#
# crt/*.c is the runtime substrate (syscalls, arena, logging, crt0).
# runtime/runtime/*.c is the ML layer (kernels, tensors, tests). Everything
# else in the tree is header-only and becomes part of these TUs.

file(GLOB SPEKI_CRT_SOURCES     "${CMAKE_CURRENT_SOURCE_DIR}/runtime/crt/*.c")
file(GLOB SPEKI_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime/*.c")

if(NOT SPEKI_CRT_SOURCES)
  message(FATAL_ERROR "no crt sources found under runtime/crt/")
endif()

add_executable(speki
  ${SPEKI_CRT_SOURCES}
  ${SPEKI_RUNTIME_SOURCES}
)

# ─── Toolchain ──────────────────────────────────────────────────────────────
#
# Freestanding means no hosted startup files and no default libraries, so we
# must be explicit about everything.

speki_apply_toolchain(speki)

# Consumers include as "runtime/runtime/kernels.h", so the project root is the
# include root (not runtime/, which would only resolve bare header names).
target_include_directories(speki PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Kernels reach exp through mathf.h, never libm. This define lets headers
# assert that at compile time, and selects the freestanding paths (e.g. our
# own struct timespec in raw_syscalls.h). The host test target does NOT define
# it, so that header takes its hosted branch and includes <time.h> instead.
target_compile_definitions(speki PRIVATE __speki_freestanding__=1)

speki_apply_flags(speki)

# No libc to link. CMake would otherwise add -lm / -lc in some configurations.
target_link_libraries(speki PRIVATE)

# Binary lands in build/ (not build/cmake/) so it sits beside build/legacy/
# from the old build.sh path without either overwriting the other.
set_target_properties(speki PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
)

# ─── Host-side test harness ─────────────────────────────────────────────────
#
# The kernels are pure computation, so they can also be compiled against the
# host libm and checked for numerical error. The freestanding binary has no
# libm to compare against, so this is the only place kernel accuracy can
# actually be verified — it is what catches a kernel that is self-consistently
# wrong (which is exactly how the softmax hsum8/hmax8 bug survived).
#
# EXCLUDE_FROM_ALL: it needs a hosted toolchain and libm, which the main
# target deliberately does not have.

if(SPEKI_HOST_TESTS)
  add_executable(speki_kernels_host EXCLUDE_FROM_ALL
    tests/host/kernels_accuracy.c
  )
  target_include_directories(speki_kernels_host PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  # Host tests keep -march=native on purpose: they measure accuracy on THIS
  # machine, so they should use whatever this CPU really has.
  target_compile_options(speki_kernels_host PRIVATE
    -O2 -Wall -Wextra -march=native
  )
  target_link_libraries(speki_kernels_host PRIVATE)
  set_target_properties(speki_kernels_host PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
  )

  include(CTest)
  add_test(NAME kernels_accuracy COMMAND speki_kernels_host)
endif()