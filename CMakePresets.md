# CMake presets for speki
#
# Configure:  cmake -B build/cmake --preset <name>
# Build:      cmake --build build/cmake
# Test:       ctest --preset dev
#
# CMakePresets.json itself cannot carry comments (the JSON preset format has
# no comment syntax and CMake rejects extra keys), so the documentation for
# these presets lives here. See also cmake/SpekiFlags.cmake for the full list
# of tunable variables and the precedence rules.
#
# PRESETS
#   dev       RelWithDebInfo, -O2, -Werror, host tests on, symbols kept.
#             This is the one to use while working on kernels.
#   release   Release, -O3, stripped, warnings not fatal. Ships.
#   native    -march=native for THIS machine, -O3, symbols kept.
#   portable  x86-64-v2 baseline so the binary runs on older x86-64 hardware
#             than Alder Lake. Slower (no AVX2/FMA/F16C) but portable.
#   asan      Address+UB sanitizers, host tests only, for debugging kernels.
#             Note the sanitizer flags are applied to ALL targets via
#             SPEKI_EXTRA_CFLAGS, so this preset is only useful with
#             SPEKI_HOST_TESTS=ON; the freestanding binary cannot be built
#             under ASan without an ASan runtime.
#
# OVERRIDING ANY VALUE
#   Command line:  cmake -B build/cmake --preset dev -DSPEKI_ARCH=native
#   Environment:   SPEKI_ARCH=native cmake -B build/cmake --preset dev
#   The environment is read at configure time only when the cache variable is
#   not already set, so -D wins over env. Configure into a fresh build dir
#   if you want the environment to take effect over an earlier -D.
#
# THE ARCH TRAP — read before using --preset native or setting SPEKI_ARCH
#   Consumer Alder Lake has NO AVX-512, NO VNNI and NO AMX silicon. The
#   default arch is 'alderlake' for that reason. -march=native on this
#   machine is safe: it resolves to exactly what the CPU supports. But
#   building with an arch that DOES have AVX-512 (skylake-avx512,
#   sapphirerapids, znver4, ...) and running the result here WILL SIGILL at
#   startup. The build cannot detect the mismatch — only running it can.
#
then build with:                           cmake --build build/cmake

Any cache variable here can also be overridden on the command line
(-DSPEKI_ARCH=native) or via the environment (SPEKI_ARCH=native).
See cmake/SpekiFlags.cmake for the full list of knobs.

IMPORTANT — the arch trap:
  'alderlake' is correct for consumer Alder Lake, which has NO AVX-512,
  NO VNNI and NO AMX silicon. Building with -march=native on such a
  machine is safe and picks up exactly what the CPU supports. But
  building with an arch that DOES have AVX-512 (skylake-avx512,
  sapphirerapids, ...) and then running the result here WILL SIGILL.
  The build cannot detect a mismatch; only the runner can.

Quick reference:
  cmake -B build/cmake --preset dev && cmake --build build/cmake
  cmake -B build/cmake --preset release && cmake --build build/cmake
  SPEKI_ARCH=native SPEKI_WERROR=ON cmake -B build/cmake --preset dev
