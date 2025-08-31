import { NextRequest, NextResponse } from "next/server";
import { backendUrl } from "@/lib/api";
import { ACCESS_COOKIE, REFRESH_COOKIE, TENANT_COOKIE } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const { username, password, tenant_id } = body || {};
  if (!username || !password) return NextResponse.json({ error: "Missing credentials" }, { status: 400 });

  // FastAPI typical token endpoint expects x-www-form-urlencoded
  const form = new URLSearchParams();
  form.set("username", username);
  form.set("password", password);

  const resp = await fetch(backendUrl("/auth/token"), {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    return NextResponse.json({ error: data?.detail ?? "Login failed" }, { status: resp.status });
  }

  const access = data.access_token || data.token || null;
  const refresh = data.refresh_token || null;
  if (!access) return NextResponse.json({ error: "No access token returned" }, { status: 500 });

  const res = NextResponse.json({ access_token: access });
  const secure = true;
  const common = { httpOnly: true, secure, sameSite: "lax" as const, path: "/" };

  res.cookies.set(ACCESS_COOKIE, access, common);
  if (refresh) res.cookies.set(REFRESH_COOKIE, refresh, common);
  if (tenant_id) res.cookies.set(TENANT_COOKIE, tenant_id, common);

  return res;
}
