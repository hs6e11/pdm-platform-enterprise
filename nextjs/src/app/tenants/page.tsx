"use client";

import Protected from "@/components/Protected";
import TenantSwitcher from "@/components/TenantSwitcher";
import { useTenant } from "@/hooks/useTenant";

export default function TenantsPage() {
  const { list } = useTenant();

  return (
    <Protected>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">Tenants</h1>
        <TenantSwitcher />
      </div>
      <div className="rounded-2xl border bg-white p-4">
        <div className="text-sm text-gray-600">Available tenants:</div>
        <ul className="mt-2 list-disc pl-6">
          {list.map((t) => <li key={t.id}>{t.name ?? t.id}</li>)}
        </ul>
      </div>
    </Protected>
  );
}
