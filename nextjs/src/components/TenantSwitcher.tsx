"use client";

import { useEffect } from "react";
import { useTenant } from "@/hooks/useTenant";

export default function TenantSwitcher() {
  const { list, current, fetchTenants, setTenant, loading } = useTenant();

  useEffect(() => { fetchTenants(); }, [fetchTenants]);

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-gray-600">Tenant:</label>
      <select
        className="border rounded-md px-2 py-1"
        value={current ?? ""}
        onChange={(e) => setTenant(e.target.value)}
      >
        <option value="">Select…</option>
        {list.map((t) => (
          <option key={t.id} value={t.id}>{t.name ?? t.id}</option>
        ))}
      </select>
      {loading && <span className="text-xs text-gray-400">Loading…</span>}
    </div>
  );
}
