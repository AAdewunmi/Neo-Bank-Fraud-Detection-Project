.PHONY: setup train_cat train_fraud score_csv test serve

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && cp -n .env.example .env || true

train_cat:
	. .venv/bin/activate && python ml/training/train_categorisation.py --input data/sample_transactions.csv --target_col category --text_cols merchant description --registry model_registry.json

train_fraud:
	. .venv/bin/activate && python ml/training/train_fraud_baseline.py --input data/sample_transactions.csv --amount_col amount --registry model_registry.json

score_csv:
	. .venv/bin/activate && PYTHONPATH=. python -m ml.inference.score_csv --input data/sample_transactions.csv --output reports/sample_scored.csv --dataset ledgerguard --registry model_registry.json

score_paysim:
	. .venv/bin/activate && PYTHONPATH=. python -m ml.inference.score_csv --input data/paysim.csv --output reports/paysim_scored.csv --dataset paysim --registry model_registry.json

test:
	. .venv/bin/activate && coverage run -m pytest -q && coverage report

serve:
	. .venv/bin/activate && python manage.py migrate && python manage.py runserver
Path: README.md (replace 
