.PHONY: lint test ci-test

lint:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests/ -q

ci-test: lint test
