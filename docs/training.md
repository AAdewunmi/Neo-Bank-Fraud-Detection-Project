# Training Guide — LedgerGuard

This document describes how to train, evaluate, and score models in the LedgerGuard project.

---

## Environment Setup

Create and activate a virtual environment, then install dependencies.

Use the Makefile `setup` target for a clean bootstrap.

---

## Categorisation Training

Trains a transaction categorisation model using text features.

- Input  
  CSV with merchant and description columns

- Output  
  Model artefact saved under `artefacts/`  
  Registry updated in `model_registry.json`

---

## Fraud Training

Two modes are supported.

### Synthetic Mode

- CI-safe and deterministic  
- Uses amount-based synthetic labels  
- Intended for engineering validation

### PaySim Mode

- Offline experimentation only  
- Uses PaySim ground truth fraud labels  
- Intended for realistic PR-AUC evaluation

---

## Scoring

CSV scoring uses the inference pipeline with strict schema checks.

LedgerGuard schema is required for dashboard ingestion.

---

## Testing

All unit and integration tests must pass before merging.

Recommended command:

- coverage run -m pytest -q

---

## Registry

The registry tracks:

- Model versions  
- Feature schema hashes  
- Metrics  
- Dataset provenance

The latest model pointer is used by default for inference.
