# CLAUDE.md — Instructions for Claude Code

Claude Code should be used as the primary repo-wide coding agent.

## Your role

You are the senior implementation agent. You should read `AGENTS.md`, respect the roadmap, implement one milestone at a time, run tests, and update docs.

## Working style

Before coding, produce a short plan:

1. files you will edit,
2. functions/classes you will add,
3. commands you will run,
4. expected outputs.

After coding, produce:

1. summary of changes,
2. tests/commands run,
3. known limitations,
4. next recommended step.

## Main tasks for Claude Code

Use Claude Code for:

- creating the full repository skeleton,
- implementing multi-file data pipeline,
- refactoring feature modules,
- wiring model training to inference,
- building FastAPI service,
- building Streamlit dashboard,
- writing docs and README,
- fixing failing tests.

## Guardrails

- Do not invent final model metrics. Use placeholders until experiments run.
- Do not add advanced infra before the baseline works.
- Do not make random train/test splits for final reports.
- Do not leave notebook-only code as the main implementation.
- Ask for confirmation before deleting files or changing project structure significantly.
