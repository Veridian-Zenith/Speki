# Contributing

Rules are short - speki is an inference runtime, not a web toy.

## Before code

- Open an issue for anything larger than a typo. State intent and the affected
  [roadmap](README.md#roadmap) item.
- One idea per PR.
- Read [docs/architecture.md](docs/architecture.md) before touching the crt or
  the kernels. Both have non-obvious constraints that look like mistakes.

## Must pass before opening PR

```sh
cmake -B build/cmake --preset dev
cmake --build build/cmake
./build/speki                      # 49 freestanding tests, exit status is the gate

cmake --build build/cmake --target speki_kernels_host
./build/speki_kernels_host         # 15 accuracy checks vs an fp128 reference
ctest --preset dev

clang-tidy -p build/cmake runtime/crt/*.c runtime/runtime/*.c
```

CI runs the same four gates. All must be green.

If you touched a kernel, paste the measured `max_rel_err` from the host suite.
"Tests pass" is not evidence of correctness here - see below.

## The rule that matters most

**A kernel that is self-consistently wrong will pass its own tests.**

This is not hypothetical. `softmax_f32` shipped with `hsum8` where it needed
`hmax8`, computing the row *sum* as the maximum. Its test passed, because the
reference implementation in the same file used the same broken polynomial. The
kernel and its reference were wrong in the same direction, so they agreed.

Consequently:

- Numerical claims come from `tests/host/kernels_accuracy.c`, which checks
  against an independent `__float128` Taylor series that shares no code and no
  constants with the implementation.
- A reference must never use the same method as the code under test. If you
  find yourself copying an algorithm into a `ref_*` function, stop.
- The freestanding binary has no libm, so it checks *structural* properties
  instead. That is deliberate: it catches shape errors, not numerical ones.
- Do not relax a tolerance to make a build green. If a kernel is wrong, fix the
  kernel.

## Style

- 4-space indent, no tabs. Comments in ASCII, short sentences.
- Comments explain *why*, not *what*. If a line needs a comment to say what it
  does, rewrite the line.
- Kernels are `static` header-only and marked `no_stack_protector` - the stack
  canary needs an `fs` base we never set up. Keep both.
- Vector loops go 8 lanes at a time with a scalar tail; the tail must be tested
  for every `n` in 1..17.
- No new dependency without an issue. The list is intentionally empty: no
  libm, no coefficient tables, no runtime. `-m<feature>` flags state a real
  requirement (see [docs/toolchain.md](docs/toolchain.md)) - do not add one
  without saying why.
- Commit messages: imperative (`Fix softmax max reduction`), not `fixed`.
  Say what was wrong and how you found it. Several commits here exist purely to
  record a disproved theory - keep those.

## Reporting

- Bugs with repro: issue with exact commands and output.
- Security: [SECURITY.md](SECURITY.md) private channel - never a public issue.
- Contact: [daedaevibin@ik.me](mailto:daedaevibin@ik.me) |
  [@daedaevibin:matrix.org](https://matrix.to/@daedaevibin:matrix.org)

## Licensing

By contributing you agree your work is under `OSL-3.0` ([LICENSE](LICENSE)).