# Data Contract v1 (Week 1)

This contract is enforced by code. If a CSV violates the contract, ingestion must fail fast with a clear error.

## Required columns (names fixed)
- timestamp: ISO 8601 string. Must include timezone (preferred UTC `Z`).
- amount: numeric, >= 0.0 (coerce; invalid -> 0.0 with warning later)
- customer_id: string, non-empty
- merchant: string (nullable -> empty string)
- description: string (nullable -> empty string)

## Optional columns (Week 1)
- category: string (only needed to train the categorisation baseline)

## Null policy
- timestamp: disallowed
- amount: coerce to float; invalid -> 0.0
- customer_id: disallowed (empty -> fail)
- merchant/description: empty string allowed

## Timezone policy
- timestamp must include timezone. Storage assumes UTC semantics.

## Extra columns
- permitted but ignored for Week 1.
- ingestion must not crash due to extra columns.

## Versioning
- Changes to required columns bump contract version and must update tests.