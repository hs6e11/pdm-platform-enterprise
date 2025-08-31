"use client";

import { create } from "zustand";
import { jwtDecode } from "jwt-decode";
import type { JwtPayload } from "@/lib/auth";

type State = {
  token: string | null;
  payload: JwtPayload | null;
  role: string | null;
  loading: boolean;
};

type Actions = {
  setToken: (token: string | null) => void;
  refresh: () => Promise<boolean>;
  logout: () => Promise<void>;
  login: (username: string, password: string, tenant_id?: string) => Promise<boolean>;
};

export const useAuth = create<State & Actions>((set, get) => ({
  token: null,
  payload: null,
  role: null,
  loading: false,
  setToken: (token) => {
    let payload: JwtPayload | null = null;
    try { payload = token ? jwtDecode(token) : null; } catch {}
    set({ token, payload, role: payload?.role ?? null });
  },
  refresh: async () => {
    try {
      const resp = await fetch("/api/refresh", { method: "POST" });
      if (resp.ok) {
        const data = await resp.json();
        if (data.access_token) get().setToken(data.access_token);
        return true;
      }
    } catch {}
    return false;
  },
  logout: async () => {
    await fetch("/api/logout", { method: "POST" });
    set({ token: null, payload: null, role: null });
    window.location.href = "/login";
  },
  login: async (username, password, tenant_id) => {
    set({ loading: true });
    try {
      const resp = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, tenant_id })
      });
      set({ loading: false });
      if (!resp.ok) return false;
      const data = await resp.json();
      if (data.access_token) get().setToken(data.access_token);
      window.location.href = "/dashboard";
      return true;
    } catch {
      set({ loading: false });
      return false;
    }
  }
}));
