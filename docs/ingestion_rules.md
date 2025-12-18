# Ingestion Rules v1 (Week 1)

Purpose: enforce the data contract with predictable coercions.

## Rejections (hard fail)
- missing any required column
- empty customer_id in any row

## Coercions (soft handling)
- amount: to numeric, invalid -> 0.0
- merchant/description: null -> empty string
- extra columns: allowed and ignored in Week 1
