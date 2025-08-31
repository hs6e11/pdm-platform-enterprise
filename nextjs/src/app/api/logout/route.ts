import { NextRequest, NextResponse } from "next/server";
import { ACCESS_COOKIE, REFRESH_COOKIE, TENANT_COOKIE } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const res = NextResponse.json({ ok: true });
  const opts = { httpOnly: true, secure: true, sameSite: "lax" as const, path: "/" };
  res.cookies.set(ACCESS_COOKIE, "", { ...opts, maxAge: 0 });
  res.cookies.set(REFRESH_COOKIE, "", { ...opts, maxAge: 0 });
  res.cookies.set(TENANT_COOKIE, "", { ...opts, maxAge: 0 });
  return res;
}
