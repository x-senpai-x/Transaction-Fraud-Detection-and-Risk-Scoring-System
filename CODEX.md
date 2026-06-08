# CODEX.md — Instructions for Codex/Codex CLI

Use Codex for focused implementation, tests, debugging, and code review.

## Best Codex tasks

- Implement a single module from a clear spec.
- Add unit tests for a module.
- Debug a stack trace.
- Improve typing and error handling.
- Generate CLI commands.
- Review code for leakage or metric mistakes.
- Convert notebook logic into production Python functions.

## Recommended Codex prompt pattern

```
Read AGENTS.md and the relevant file in agents/.
Implement only this module: <module name>.
Do not change unrelated files.
Add tests.
Run pytest for affected tests.
Summarize changes and limitations.
```

## Codex guardrails

- Keep changes small and reviewable.
- Do not refactor the entire repo unless explicitly asked.
- Do not install unnecessary libraries.
- Do not replace LightGBM/XGBoost with deep learning unless asked.
- Do not chase marginal metric improvements before baseline is done.

codex resume 019ea388-6eb4-7232-8c83-26a27c4afc15