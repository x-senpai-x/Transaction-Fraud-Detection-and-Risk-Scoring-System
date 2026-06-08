# Prompt — Architect Review

Use this prompt at the beginning of a coding session.

```txt
You are helping me build Transaction Risk Engine, an adaptive transaction risk engine for a resume project targeting ML/DS/SDE internships.

Read these files first:
- AGENTS.md
- docs/PROJECT_SCOPE.md
- docs/IMPLEMENTATION_ROADMAP.md
- docs/DATASET_AND_LEAKAGE_RULES.md
- docs/ML_CONCEPTS_TARGETED.md
- docs/reference/real_time_fraud_detection_system_design.md

Note: `docs/reference/real_time_fraud_detection_system_design.md` is an in-depth design document and is enterprise grade. We are not targeting that level of complexity, but it is a good reference for understanding the problem space and potential features. We are only targeting a baseline model that I can add to my resume for an on-campus internship.

Do not code yet.
Give me:
1. the current milestone we should implement,
2. exact files to create/edit,
3. risks/leakage traps,
4. tests to add,
5. commands to run.

Keep scope controlled. Baseline first, advanced features later.
```
