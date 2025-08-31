import { NextRequest, NextResponse } from "next/server";
import { backendUrl } from "@/lib/api";
import { ACCESS_COOKIE, TENANT_COOKIE } from "@/lib/auth";

export async function GET(req: NextRequest, { params }: { params: { path: string[] } }) {
  return handleProxy(req, params);
}
export async function POST(req: NextRequest, { params }: { params: { path: string[] } }) {
  return handleProxy(req, params);
}
export async function PUT(req: NextRequest, { params }: { params: { path: string[] } }) {
  return handleProxy(req, params);
}
export async function PATCH(req: NextRequest, { params }: { params: { path: string[] } }) {
  return handleProxy(req, params);
}
export async function DELETE(req: NextRequest, { params }: { params: { path: string[] } }) {
  return handleProxy(req, params);
}

async function handleProxy(req: NextRequest, { path }: { path: string[] }) {
  const targetPath = "/" + (path?.join("/") ?? "");
  const url = new URL(req.url);
  const target = backendUrl(`${targetPath}${url.search}`);

  const access = req.cookies.get(ACCESS_COOKIE)?.value;
  const tenant = req.cookies.get(TENANT_COOKIE)?.value;

  const headers: Record<string, string> = {};
  req.headers.forEach((v, k) => {
    if (!["host", "accept-encoding", "content-length"].includes(k.toLowerCase())) headers[k] = v;
  });
  if (access) headers["Authorization"] = `Bearer ${access}`;
  if (tenant) headers["X-Tenant-Id"] = tenant;

  const init: RequestInit = {
    method: req.method,
    headers,
    body: req.method !== "GET" && req.method !== "HEAD" ? await req.arrayBuffer() : undefined,
    redirect: "manual",
  };

  const resp = await fetch(target, init);
  const body = await resp.arrayBuffer();
  const res = new NextResponse(body, { status: resp.status, headers: resp.headers });
  return res;
}
