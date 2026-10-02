# SPDX-License-Identifier: OSL-3.0
# CMakeLists.txt — Speki build.
#
# Produces a single fully-static freestanding ELF: `speki`.
# No libc, no libgcc, no compiler-rt. Entry point is our own `_start`.
#
# Requires: clang (or gcc), lld, cmake, ninja.
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/speki

cmake_minimum_required(VERSION 3.20)
project(speki C)

# ─── Toolchain ─────────────────────────────────────────────────────────────
#
# Freestanding means no hosted startup files and no default libraries, so we
# have to be explicit about everything. We link with lld because the default
# GNU ld pulls in behavior we do not want (and is slower to link).

if(NOT CMAKE_C_COMPILER MATCHES "clang")
  message(WARNING "speki is built for clang; got '${CMAKE_C_COMPILER}'.")
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Sources: crt/*.c (runtime substrate) plus runtime/runtime/*.c (ML layer).
# Everything else in the tree is header-only and becomes part of these TUs.
file(GLOB SPEKI_CRT_SOURCES     "${CMAKE_CURRENT_SOURCE_DIR}/runtime/crt/*.c")
file(GLOB SPEKI_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime/*.c")

if(NOT SPEKI_CRT_SOURCES)
  message(FATAL_ERROR "no crt sources found under runtime/crt/")
endif()

add_executable(speki
  ${SPEKI_CRT_SOURCES}
  ${SPEKI_RUNTIME_SOURCES}
)

# ─── Target flags ──────────────────────────────────────────────────────────
#
# -march=alderlake gives AVX2 + FMA + F16C + BMI1/2 + ADX + GFNI + VAES +
# VPCLMULQDQ. Consumer Alder Lake has no AVX-512/VNNI/AMX silicon, so we do
# not enable -mavx512f and friends: they would compile and then SIGILL.
#
# Security/safety flags mirror the project's fish config: branch-only CFG
# (-fcf-protection=branch, not =full — we have no shadow stack and never
# enable CET), stack-clash protection, zero-init locals.
#
# -ffreestanding: no hosted environment assumptions. -nostdlib/-nostartfiles at
# link time do the real work.

target_compile_options(speki PRIVATE
  -march=alderlake -mtune=alderlake
  -mavx2 -mfma -mf16c -mbmi -mbmi2 -madx -mgfni -mvaes -mpclmul
  -O3
  -fno-plt -fno-rtti -fno-exceptions
  -fno-unwind-tables -fno-asynchronous-unwind-tables
  -fmerge-all-constants
  -ftrivial-auto-var-init=zero
  -fstack-clash-protection
  -fcf-protection=branch
  -ffunction-sections -fdata-sections
  -ffreestanding
  -Wall -Wextra -Wno-unused-parameter
)

target_compile_definitions(speki PRIVATE
  # Kernels call expf/tanhf indirectly via mathf.h; nothing here may reach
  # libm, so make an accidental <math.h> include a hard error.
  SPEKI_FREESTANDING=1
)

target_link_options(speki PRIVATE
  -static
  -fuse-ld=lld
  -nostdlib -nostartfiles
  -Wl,--gc-sections
  -Wl,--exclude-libs=ALL
  -Wl,-z,now -z,relro -z,noseparate-code
  -Wl,--build-id=none
  -Wl,--strip-all
)

# No libc, so there is nothing to link. CMake adds -lm / -lc by default in
# some configurations; strip those explicitly.
target_link_libraries(speki PRIVATE)

set_target_properties(speki PROPERTIES
  LINK_FLAGS_RELEASE ""
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ─── Host-side test harness ────────────────────────────────────────────────
#
# The kernels are pure computation, so they can also be compiled and checked
# against the host libm. This target is the ground truth for kernel accuracy:
# `ctest` runs it, and any kernel whose error exceeds what mathf.h claims will
# fail there rather than silently shipping.
#
# It is NOT part of `all` by default because it needs libm and a hosted
# toolchain; build it with: cmake --build build --target speki_kernels_host

option(SPEKI_BUILD_HOST_TESTS "Build host-side kernel accuracy tests (needs libm)" ON)

if(SPEKI_BUILD_HOST_TESTS)
  add_executable(speki_kernels_host EXCLUDE_FROM_ALL
    tests/host/kernels_accuracy.c
  )
  target_include_directories(speki_kernels_host PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
  )
  target_compile_options(speki_kernels_host PRIVATE
    -O2 -Wall -Wextra -march=native
  )
  target_link_libraries(speki_kernels_host PRIVATE m)

  enable_testing()
  add_test(NAME kernels_accuracy COMMAND speki_kernels_host)
endif()