import { NextRequest, NextResponse } from "next/server";

export const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
export const BACKEND_REFRESH_PATH = process.env.BACKEND_REFRESH_PATH || "";

export function backendUrl(path: string) {
  return `${BACKEND_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

// Helper for route handlers
export async function proxyRequest(req: NextRequest, path: string) {
  const url = new URL(req.url);
  const target = backendUrl(`${path}${url.search}`);
  const init: RequestInit = {
    method: req.method,
    headers: {
      // Copy headers except host and accept-encoding
      ...Object.fromEntries(
        Array.from(req.headers.entries()).filter(([k]) => !["host", "accept-encoding", "content-length"].includes(k.toLowerCase()))
      ),
    },
    body: req.method !== "GET" && req.method !== "HEAD" ? await req.arrayBuffer() : undefined,
    redirect: "manual",
  };
  const resp = await fetch(target, init);
  const buf = await resp.arrayBuffer();
  const res = new NextResponse(buf, {
    status: resp.status,
    headers: resp.headers,
  });
  return res;
}
