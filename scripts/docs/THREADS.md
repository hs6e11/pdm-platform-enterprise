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
