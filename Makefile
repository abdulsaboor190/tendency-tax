.PHONY: setup data test dev-run confirm app diagnostics power secondary download

setup:
	uv sync

download:
	uv run python -m scripts.download_data

data:
	uv run python -m src.load

diagnostics:
	uv run python -m src.diagnostics

power:
	uv run python -m src.simulate

secondary:
	uv run python -m scripts.run_all --stage secondary

test:
	uv run pytest
	uv run ruff check .

dev-run:
	uv run python -m scripts.run_all --stage dev

confirm:
	uv run python -m scripts.run_all --stage confirm

app:
	uv run python -m streamlit run app/app.py
