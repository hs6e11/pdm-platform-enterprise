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
