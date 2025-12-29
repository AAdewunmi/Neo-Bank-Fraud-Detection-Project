# tests/test_schema_hash.py
from ml.training.utils import schema_hash


def test_schema_hash_is_stable_for_same_inputs() -> None:
    h1 = schema_hash(["merchant", "description", "category"])
    h2 = schema_hash(["merchant", "description", "category"])
    assert h1 == h2


# def test_schema_hash_changes_when_columns_change() -> None:
#     base = schema_hash(["merchant", "description", "category"])
#     changed = schema_hash(["merchant", "description", "category", "amount"])
#     assert base != changed


# def test_schema_hash_trims_and_normalises_columns() -> None:
#     h1 = schema_hash([" Merchant ", "Description", "category"])
#     h2 = schema_hash(["merchant", "description", "category"])
#     assert h1 == h2
