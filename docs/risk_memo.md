# Risk Memo — Neo-bank MVP (v0.3)

## Purpose

Summarise the current fraud flagging strategy used in the LedgerGuard MVP, including threshold selection, operational trade-offs, and planned controls.  
This document supports engineering review, risk sign-off, and portfolio evaluation.

---

## Current Models

- **Categorisation**  
  Latest version recorded in `model_registry.json`  
  MiniLM embeddings with LightGBM, with TF-IDF plus linear model as a fallback

- **Fraud**  
  Latest version recorded in `model_registry.json`  
  Isolation Forest for baseline experiments  
  XGBoost for supervised fraud detection

Offline evaluation uses PaySim where available.  
Online dashboard scoring accepts only the LedgerGuard transaction schema.

---

## Business Threshold

- **Proposed threshold**  
  0.65

- **Rationale**  
  Selected from the precision–recall curve to prioritise precision during early analyst workflows, where false positives are more costly than missed low-value fraud.

- **Observed metrics at threshold 0.65**  
  Precision@0.65: populated from latest fraud metrics JSON  
  Recall@0.65: populated from latest fraud metrics JSON

Thresholds are reviewed whenever the fraud model, feature schema, or dataset changes.

---

## Impact of Errors

- **False positives**  
  Increased analyst review time  
  Potential customer trust erosion

- **False negatives**  
  Direct financial loss  
  Missed emerging fraud patterns

The current threshold favours analyst efficiency and trust during MVP stages.

---

## Controls

- Deterministic feature schema with enforced train–inference parity  
- Versioned models and rollback via `model_registry.json`  
- Threshold tables persisted per model version  
- Planned rules overlay for known risky merchants  
- Planned analyst feedback export loop

---

## Monitoring (Next)

- Percentage of transactions flagged over time  
- Drift in engineered features, especially `amount_z`  
- Precision on adjudicated cases when analyst labels are available
