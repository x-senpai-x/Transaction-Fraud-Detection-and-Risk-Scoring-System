# Prompt — Feature Exploration

Use this after baseline features and model are working.

```txt
You are a feature engineering researcher for Transaction Risk Engine.

Read:
- docs/DATASET_AND_LEAKAGE_RULES.md
- docs/ML_CONCEPTS_TARGETED.md
- agents/02_feature_engineering_agent.md
- agents/06_graph_features_agent.md
- docs/reference/real_time_fraud_detection_system_design.md

Note: `docs/reference/real_time_fraud_detection_system_design.md` is an in-depth design document and is enterprise grade. We are not targeting that level of complexity, but it is a good reference for understanding the problem space and potential features. We are only targeting a baseline model that I can add to my resume for an on-campus internship.

Goal:
Suggest 10 candidate features that could improve fraud detection without target leakage.

For each feature, provide:
1. feature name,
2. intuition,
3. exact columns needed,
4. whether it is past-only/time-aware,
5. implementation complexity,
6. expected risk of leakage,
7. how to test whether it helps.

Do not implement yet. Output a prioritized list.
```
