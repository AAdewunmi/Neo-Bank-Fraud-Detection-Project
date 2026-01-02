# Fraud Model Card — LedgerGuard

## Model Overview

- **Model type**  
  Supervised XGBoost classifier

- **Purpose**  
  Estimate transaction-level fraud risk for analyst triage

- **Output**  
  Continuous risk score in the range 0 to 1

---

## Training Data

- **Primary dataset**  
  LedgerGuard synthetic dataset for CI-safe training  
  PaySim dataset for offline evaluation

- **Label source**  
  Synthetic amount-based labels for CI  
  Ground truth `isFraud` labels for PaySim experiments

---

## Features

Ordered feature schema enforced across training and inference:

- amount  
- amount_z  
- amount_mean_cust  
- amount_std_cust  
- hour

Schema hash is stored in `model_registry.json`.

---

## Evaluation Metrics

- **Primary metric**  
  Average Precision (PR-AUC)

- **Threshold analysis**  
  Precision–recall curve and threshold table persisted per run

---

## Limitations

- No probability calibration applied yet  
- PaySim schema differs from live ingestion schema  
- Customer aggregates are approximated during batch inference

---

## Intended Use

- Analyst decision support  
- Offline experimentation and learning  
- Not suitable for automated enforcement actions

---

## Versioning and Rollback

- All artefacts are versioned and immutable  
- Rollback supported by updating `model_registry.json`
