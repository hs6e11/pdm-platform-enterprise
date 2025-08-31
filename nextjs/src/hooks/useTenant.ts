"use client";

import { create } from "zustand";

type Tenant = { id: string; name: string };

type State = {
  current: string | null;
  list: Tenant[];
  loading: boolean;
};

type Actions = {
  fetchTenants: () => Promise<void>;
  setTenant: (id: string) => Promise<void>;
};

export const useTenant = create<State & Actions>((set, get) => ({
  current: null,
  list: [],
  loading: false,
  fetchTenants: async () => {
    set({ loading: true });
    try {
      const resp = await fetch("/api/proxy/tenants");
      if (resp.ok) {
        const data = await resp.json();
        const items: Tenant[] = Array.isArray(data) ? data : (data.items || []);
        set({ list: items, loading: false });
      } else {
        set({ list: [], loading: false });
      }
    } catch {
      set({ list: [], loading: false });
    }
  },
  setTenant: async (id: string) => {
    // Store as httpOnly from server by calling a tiny endpoint
    await fetch("/api/set-tenant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant_id: id })
    });
    set({ current: id });
    // Optional: refresh the page or re-fetch data
  }
}));
