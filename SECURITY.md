# Security Policy

## Supported code

Until this repository starts publishing formal release lines, security fixes are handled on the `main` branch.

## Reporting a vulnerability

Do not open a public issue for a security problem.

Report it privately through GitHub security reporting or by contacting the repository owner directly through GitHub. Include:

- A clear description of the issue.
- Affected backend or model path.
- Required feature flags.
- Platform details.
- Reproduction steps or a minimal proof of concept.
- Any potential impact you have already confirmed.

## What to expect

- Best-effort acknowledgement within 5 business days.
- A request for more detail if the report is incomplete.
- Coordination on disclosure timing once the issue is understood.

## Scope notes

This repository loads third-party model weights and optional runtime backends. Security reports are especially useful when they involve:

- Unsafe or surprising file-resolution behavior.
- Download-path issues.
- Deserialization or model-loading vulnerabilities.
- Backend-specific crashes or memory corruption.
- Cases where user-controlled input crosses a trust boundary unexpectedly.

Model license questions are not security issues. Those should go through normal support channels.