# Agent 06 — Graph Features Optional Extension

## Mission

Add entity relationship features to make the project stand out beyond a standard fraud classifier.

## Inputs

- joined transactions with proxy IDs,
- chronological split,
- optional model predictions.

## Outputs

- graph feature table,
- optional graph visualization,
- ablation report showing whether graph features help.

## Simple graph features

Start with these before embeddings:

- `device_unique_cards`
- `card_unique_devices`
- `email_unique_cards`
- `address_unique_cards`
- `card_device_pair_count`
- `email_device_pair_count`
- `entity_degree_card`
- `entity_degree_device`
- `connected_component_size`

## Optional embedding features

Only after simple graph features work:

- Node2Vec embeddings,
- GraphSAGE experiment,
- compare LightGBM with and without embeddings.

## Leakage rule

Graph features for a transaction must be computed from past edges only if used in final evaluation.

## Acceptance criteria

- Graph features merge cleanly with baseline feature matrix.
- Ablation report exists.
- No future labels used in graph features.

## Agent prompt

```txt
Read AGENTS.md and docs/DATASET_AND_LEAKAGE_RULES.md.
Implement simple graph-derived features using proxy entities.
Do not implement GNN first.
Create ablation report comparing baseline vs baseline+graph features.
```
