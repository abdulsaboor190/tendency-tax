.PHONY: setup data test dev-run confirm app

setup:
	uv sync

data:
	@echo data target is a stub

test:
	uv run pytest
	uv run ruff check .

dev-run:
	@echo dev-run target is a stub

confirm:
	@echo confirm target is a stub

app:
	@echo app target is a stub
