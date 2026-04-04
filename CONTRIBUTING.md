# Contributing

Thanks for contributing to tts-rs.

This repository is a model-integration project as much as it is a Rust library. Good contributions keep the public API honest, keep backend behavior explicit, and avoid documenting features that only exist upstream but not in this crate yet.

## Before you open a pull request

- Keep the scope focused. Do not mix a model change, formatting sweep, and unrelated refactor in one PR.
- Update docs when you change the public surface, supported models, flags, or examples.
- If a backend does not support something yet, return an explicit error instead of silently ignoring it.
- Keep examples aligned with exported API. If an example only exercises in-tree experimental code, label it that way.

## Local workflow

Use targeted checks first:

```bash
cargo test
cargo test --test test_traits
cargo run --example generate_kokoro --release
```

Ignored integration tests usually require local model weights. Run those only when you have the assets available and want to validate end-to-end behavior.

## Coding expectations

- Match the existing builder-style configuration API.
- Preserve feature gates and avoid enabling heavyweight dependencies by accident.
- Prefer small, explicit errors over fallback magic.
- Keep unsupported model capabilities explicit in both code and docs.
- Avoid unrelated churn in generated audio paths unless the change requires it.

## Model additions

When adding a new model or variant, include all of the following in the same PR when possible:

- The loader and runtime implementation.
- File-resolution rules in `TtsConfig` and `ModelFiles`, if needed.
- At least one example or smoke test path.
- README updates covering status, pros and cons, and license notes.

## Pull request checklist

- Explain the user-visible change.
- List the feature flags or platforms you tested.
- Mention whether model assets were required to validate the change.
- Call out any upstream capability that is still intentionally unsupported in this crate.

## Security issues

Do not file public issues for vulnerabilities. Use the process in [SECURITY.md](SECURITY.md).