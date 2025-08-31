import { cookies } from "next/headers";
import { jwtDecode } from "jwt-decode";

export const ACCESS_COOKIE = "access_token";
export const REFRESH_COOKIE = "refresh_token";
export const TENANT_COOKIE = "tenant_id";

export type JwtPayload = {
  sub?: string;
  exp?: number;
  iat?: number;
  role?: string;
  tenant_id?: string;
  [key: string]: any;
};

export function getAccessTokenFromCookies() {
  return cookies().get(ACCESS_COOKIE)?.value;
}

export function decodeToken(token?: string) {
  if (!token) return null;
  try {
    return jwtDecode<JwtPayload>(token);
  } catch {
    return null;
  }
}

export function isExpired(payload: JwtPayload | null) {
  if (!payload?.exp) return false;
  return Math.floor(Date.now() / 1000) >= payload.exp;
}
