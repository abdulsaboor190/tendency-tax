.PHONY: setup data test dev-run confirm app diagnostics power

setup:
	uv sync

data:
	uv run python -m src.load

diagnostics:
	uv run python -m src.diagnostics

power:
	uv run python -m src.simulate

test:
	uv run pytest
	uv run ruff check .

dev-run:
	@echo dev-run target is a stub

confirm:
	@echo confirm target is a stub

app:
	@echo app target is a stub
