up:
	 docker compose up -d

down:
	 docker compose down

migrate:
	 alembic upgrade head

test:
	 pytest -q tests

lint:
	 ruff check api/ scripts/ tests/

dev:
	 export DATABASE_URL=postgres://postgres:password@localhost:5433/pdm_v2 && uvicorn api.main:app --reload --port 8000
