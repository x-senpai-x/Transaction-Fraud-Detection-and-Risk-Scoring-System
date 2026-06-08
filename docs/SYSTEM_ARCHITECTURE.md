# System Architecture

## Baseline architecture

```txt
IEEE-CIS CSV files
      ↓
Data ingestion and cleaning
      ↓
Time-based split
      ↓
Feature engineering
      ↓
Model training and evaluation
      ↓
Saved model + preprocessing artifacts
      ↓
FastAPI inference service
      ↓
Streamlit dashboard
      ↓
SHAP explanations and monitoring reports
```

## Production-inspired architecture, simplified

The enterprise version would use Kafka/Flink/feature stores. This project simulates the same ideas without heavy infra.

```txt
Historical transactions
      ↓
Offline feature builder
      ↓
Model training
      ↓
Saved model

New transaction
      ↓
API request
      ↓
Feature builder / feature cache
      ↓
Model inference
      ↓
Risk decision
      ↓
Dashboard + logs
```

## Core components

### Data pipeline

Loads and cleans the raw dataset, constructs proxy entity IDs, and saves clean data.

### Feature pipeline

Creates basic, frequency, historical, velocity, and optional graph features.

### Training pipeline

Trains baseline and boosted tree models, evaluates them using imbalance-aware metrics, and saves artifacts.

### Inference service

Loads the saved model and feature metadata, accepts a transaction payload, returns probability, risk score, decision, and top reasons.

### Dashboard

Shows metrics, risk score distribution, transaction explorer, and explanations.

### Monitoring

Tracks feature/prediction drift and compares champion vs challenger models.
