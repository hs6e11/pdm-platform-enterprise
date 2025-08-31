import { NextRequest, NextResponse } from "next/server";
import { BACKEND_REFRESH_PATH, backendUrl } from "@/lib/api";
import { ACCESS_COOKIE, REFRESH_COOKIE } from "@/lib/auth";

export async function POST(req: NextRequest) {
  if (!BACKEND_REFRESH_PATH) {
    // Try to reuse current access token from cookie (e.g., page reload)
    const access = req.cookies.get(ACCESS_COOKIE)?.value;
    if (access) return NextResponse.json({ access_token: access }, { status: 200 });
    return NextResponse.json({ error: "Refresh not configured" }, { status: 501 });
  }

  const refresh = req.cookies.get(REFRESH_COOKIE)?.value;
  if (!refresh) return NextResponse.json({ error: "No refresh token" }, { status: 401 });

  const resp = await fetch(backendUrl(BACKEND_REFRESH_PATH), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh })
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) return NextResponse.json({ error: data?.detail ?? "Refresh failed" }, { status: resp.status });

  const access = data.access_token || data.token || null;
  if (!access) return NextResponse.json({ error: "No access token" }, { status: 500 });

  const res = NextResponse.json({ access_token: access });
  const common = { httpOnly: true, secure: true, sameSite: "lax" as const, path: "/" };
  res.cookies.set(ACCESS_COOKIE, access, common);
  return res;
}
