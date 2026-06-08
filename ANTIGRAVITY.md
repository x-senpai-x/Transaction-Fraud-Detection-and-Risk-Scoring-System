# ANTIGRAVITY.md — Optional Agent Instructions

Use Antigravity only if you have access and it fits your workflow. The project must not depend on it.

## Good Antigravity uses

- Explore UI/dashboard flows.
- Run browser-based checks on the Streamlit/FastAPI demo.
- Suggest feature ideas after reading EDA outputs.
- Test end-to-end flows: upload sample transaction, call API, view dashboard.
- Explore graph/relationship visualizations.

## Do not use Antigravity for

- uncontrolled repo-wide edits,
- changing ML methodology without approval,
- adding huge infrastructure,
- replacing the baseline plan.

## Best prompt

```
You are exploring the Transaction Risk Engine project as a QA/product agent.
Do not edit core ML code.
Run the app, inspect the dashboard/API behavior, identify confusing UX, missing docs, or broken flows.
Suggest improvements in a short report.
```
