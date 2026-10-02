# Security Policy

speki is a freestanding inference runtime: no libc, no dynamic loader, static
ELF. Most of the usual attack surface does not exist. What remains is
unusually unforgiving, because there is no runtime underneath to catch
anything - see "Failure modes" below.

## How to report

Use GitHub's private security advisory (Security tab -> Report a vulnerability)
for `Veridian-Zenith/Speki`. If you cannot, reach out via:

- **Email:** [daedaevibin@ik.me](mailto:daedaevibin@ik.me)
- **Matrix:** [@daedaevibin:matrix.org](https://matrix.to/@daedaevibin:matrix.org)

Do not open a public issue for anything exploitable. Do not disclose
externally until a fix is released.

## What to expect

- Confirmation within 48 hours.
- Fix or explicit wont-fix rationale before any disclosure.
- Credit if requested.

## In scope

- Memory corruption reachable from model weights, a weight file, or tokenizer
  input. The arena allocator has no `free`; a length field that is trusted
  before it is validated is the obvious class of bug.
- Out-of-bounds read or write in any kernel, including the scalar tail paths
  that only run for `n % 8 != 0`.
- Integer overflow in a tensor shape or byte-cap computation.
- Anything that makes a *silently wrong* numerical result rather than a crash.
  See the next section - this is the one that matters here.
- Build or CI compromise: a workflow that fetches unpinned code, or a pinned
  key that is not actually verified.

## Out of scope

- Denial of service from a malformed file on a machine you already control.
- Performance regressions.
- Missing hardening features on a binary that is already static and
  freestanding. There is no dynamic loader to hijack.

## Failure modes that are security-relevant here

speki has no signal handlers, no allocator, and no libc. A kernel bug does not
raise a catchable error; it produces one of:

- `#GP` on a misaligned aligned-store, which surfaces as SIGILL.
- A write below the stack guard page, SIGSEGV.
- A silently incorrect float.

That last one is the dangerous one and the reason
`tests/host/kernels_accuracy.c` exists. A kernel that is wrong by 1e-6
relative does not crash. Every downstream operation in a transformer is
scale-invariant, so the error propagates silently through every weight and
emerges as a model that is subtly degraded with no indication of why. Treat
"kernel is numerically wrong" as a security issue, not just a correctness one.

## Hardening posture

What is already in place, so reports do not duplicate work:

- `-static -nostdlib -nostartfiles`: no interpreter, no dynamic loader, no
  shared library dependencies. CI asserts this on every push by checking for
  `NEEDED` entries and undefined symbols.
- No stack canary (`no_stack_protector`): the canary needs an `fs` base that is
  never established, so enabling it would read garbage and abort. Documented in
  `docs/architecture.md`.
- `-fcf-protection=branch`, not `=full`: CET is not enabled, so shadow-stack
  landing pads would be theatre.
- `-ftrivial-auto-var-init=zero`: no reliance on indeterminate locals.
- `-fstack-clash-protection`.
- Full RELRO (`-z now -z relro`) even though there is nothing dynamic to
  relocate.
- `apt.llvm.org` signing key verified against a pinned fingerprint in CI.

## Disclosure

Coordinated disclosure. Fix first, then release, then credit unless the
reporter prefers otherwise.