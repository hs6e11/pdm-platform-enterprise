#!/usr/bin/env bash
set -euo pipefail

echo "==> Bootstrapping project coordination files ..."

# Ensure dirs
mkdir -p docs/DECISIONS
mkdir -p interfaces/schemas
mkdir -p .github
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p scripts

# 1) docs/THREADS.md
cat > docs/THREADS.md <<'MD'
# Project Threads Directory

- **Backend/API** (this chat)
  - Scope: FastAPI, JWT/RBAC, tenants, ingest, Timescale, rate-limit, audit, tests
  - Contracts touched: interfaces/openapi.yaml, interfaces/schemas/*
  - Last updated: 2025-08-30

- **DevOps & Observability**
  - Scope: docker-compose, Prometheus, Grafana, alerting
  - Contracts touched: none (reads /metrics)

- **IoT Edge Gateway**
  - Scope: agent buffering/retry, config, MQTT
  - Contracts touched: interfaces/openapi.yaml (ingest endpoints)

- **Frontend (Next.js)**
  - Scope: auth UI, dashboards, charts, audit viewer
  - Contracts touched: interfaces/openapi.yaml

- **Data & ML**
  - Scope: continuous aggregates, anomaly detection, analytics endpoints
  - Contracts touched: interfaces/openapi.yaml, interfaces/schemas/analytics/*

- **CI/CD**
  - Scope: GH Actions, migrations, tests
  - Contracts touched: none
MD

# 2) docs/HANDOFF_TEMPLATE.md
cat > docs/HANDOFF_TEMPLATE.md <<'MD'
# Handoff Note

**Module/Thread:** (e.g., Backend/API)  
**Summary of change:** (1–2 lines)  
**Breaking changes:** yes/no (explain)  

**Contracts affected:**  
- OpenAPI: interfaces/openapi.yaml  
- JSON Schemas: interfaces/schemas/<name>.json  
- DB: Alembic revision <rev> adds/changes tables  

**Required actions for other threads:**  
- Frontend: update client for /auth/token response field <x>  
- Gateway: send new field <y> in payload (schema vX.Y)  
- DevOps: new metric name <z> in /metrics  

**Testing notes:**  
- `make up && make test`  
- curl examples…
MD

# 3) docs/DECISIONS/ADR-0000-template.md
cat > docs/DECISIONS/ADR-0000-template.md <<'MD'
# ADR-0000: <Decision Title>
Date: YYYY-MM-DD
Status: Proposed | Accepted | Superseded by ADR-XXXX

## Context
(Why are we deciding this?)

## Decision
(What are we doing?)

## Consequences
(Good/bad, migration steps, affected modules)

## Links
(PRs, issues, diagrams)
MD

# 4) docs/CHANGELOG.md
cat > docs/CHANGELOG.md <<'MD'
## [Unreleased]
- Backend: added /api/iot/bulk (schema v1.1). Affects Gateway & Frontend.
- Data/ML: 5-min continuous aggregate for temperature.

## 2025-08-30
- Backend: JWT/RBAC + Timescale hypertable + rate limit + audit.
MD

# 5) interfaces/openapi.yaml
cat > interfaces/openapi.yaml <<'YML'
openapi: 3.0.3
info:
  title: PDM Platform API
  version: 2.0.0
paths:
  /auth/token:
    post:
      summary: Issue JWT
      requestBody:
        content:
          application/x-www-form-urlencoded:
            schema:
              type: object
              required: [username, password]
              properties:
                username: { type: string }
                password: { type: string }
      responses:
        "200":
          description: JWT token
  /tenants:
    get:
      summary: List tenants
      responses: { "200": { description: OK } }
    post:
      summary: Create tenant
      responses: { "200": { description: OK } }
  /api/iot/data:
    post:
      summary: Ingest one reading
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/IngestReading'
      responses:
        "200": { description: OK }
components:
  schemas:
    IngestReading:
      type: object
      required: [machine_id, timestamp, sensors]
      properties:
        machine_id: { type: string }
        timestamp: { type: string, format: date-time }
        sensors:   { type: object, additionalProperties: true }
YML

# 6) interfaces/schemas/ingest-v1.json
cat > interfaces/schemas/ingest-v1.json <<'JSON'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IngestReading v1",
  "type": "object",
  "required": ["machine_id", "timestamp", "sensors"],
  "properties": {
    "machine_id": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "sensors": {
      "type": "object",
      "additionalProperties": { "type": "number" }
    }
  }
}
JSON

# 7) .github/pull_request_template.md
cat > .github/pull_request_template.md <<'MD'
## Summary
(what changed)

## Contracts
- [ ] OpenAPI updated at interfaces/openapi.yaml
- [ ] JSON Schema(s) updated at interfaces/schemas/*
- [ ] Alembic migration added (if DB changed)

## Cross-thread Impact
- Frontend:
- Gateway:
- DevOps:
- Data/ML:
(brief actions or “none”)

## Handoff
(link/copy docs/HANDOFF_TEMPLATE.md content here)

## Test Plan
- commands / curl / screenshots
MD

# 8) Makefile (needs real TABs)
# We use a quoted heredoc and embed real tabs before the commands.
cat > Makefile <<'MAKE'
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
MAKE

echo "==> Done."
echo "Next steps:"
echo "  1) Review files under docs/, interfaces/, .github/, and Makefile"
echo "  2) git add -A && git commit -m 'chore: add cross-thread coordination files'"

