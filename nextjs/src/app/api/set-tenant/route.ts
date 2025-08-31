import { NextRequest, NextResponse } from "next/server";
import { TENANT_COOKIE } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const tenant_id = body?.tenant_id;
  if (!tenant_id) return NextResponse.json({ error: "Missing tenant_id" }, { status: 400 });
  const res = NextResponse.json({ ok: true });
  res.cookies.set(TENANT_COOKIE, tenant_id, { httpOnly: true, secure: true, sameSite: "lax", path: "/" });
  return res;
}
