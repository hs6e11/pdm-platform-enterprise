# PDM Frontend (Next.js 14 + Tailwind)

Role-based admin/operator dashboard for a FastAPI backend with JWT and multi-tenant support.

## Quick Start

```bash
cd nextjs
cp .env.example .env
# Edit BACKEND_URL in .env to point to your FastAPI (e.g., http://localhost:8000)
npm install
npm run dev
```

Open http://localhost:3000

### Login
Use any valid backend credentials. The login form posts to `/api/login`, which exchanges credentials with `${BACKEND_URL}/auth/token` and stores JWTs as **httpOnly** cookies.

If your backend exposes a refresh endpoint, set `BACKEND_REFRESH_PATH` (default `/auth/refresh`). The UI will try `/api/refresh` on load/navigation to refresh the access token.

### Proxy
All data requests should use the proxy: `/api/proxy/<backend_path>`
- Adds `Authorization: Bearer <access_token>` from cookie.
- Adds `X-Tenant-Id` from cookie if selected.
- Avoids CORS headaches.

Examples:
- Fetch tenants: `GET /api/proxy/tenants`
- Sensor data: `GET /api/proxy/api/iot/data?limit=50`
- Audit logs: `GET /api/proxy/audit/logs?limit=100`

### Tenants
Use the Tenants page to set a tenant cookie via `/api/set-tenant`. The value is also included in proxied requests via `X-Tenant-Id`.

### Live Charts
- Preferred: set `SENSOR_WS_URL` or `SENSOR_SSE_URL` in `.env` and expose a stream from your backend.
- Fallback: the page will poll `/api/iot/data` via the proxy if stream is not available.

### Roles
The dashboard includes a `RoleGuard` for client-side visibility based on the `role` claim in the JWT. **RBAC must still be enforced on the backend**.

## Docker

Build and run:
```bash
docker build -t pdm-frontend .
docker run --rm -p 3000:3000 --env BACKEND_URL=http://host.docker.internal:8000 pdm-frontend
```

## Structure

- `src/app` — Next.js App Router pages & API routes
- `src/components` — UI building blocks
- `src/hooks` — auth/tenant/live helpers
- `src/lib` — small utilities
- `src/app/api/proxy` — generic auth-aware reverse proxy

## Notes

- Cookies are `httpOnly`, `SameSite=Lax`, `Secure` (set to true by default).
- If you do not have a refresh endpoint, the app will still work: tokens are read on each load; on expiry the user is redirected to `/login`.
