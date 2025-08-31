"use client";

import Protected from "@/components/Protected";
import RoleGuard from "@/components/RoleGuard";
import TenantSwitcher from "@/components/TenantSwitcher";
import dynamic from "next/dynamic";
const ChartCard = dynamic(() => import("@/components/ChartCard"), { ssr: false });
import { useEffect, useState } from "react";


type Point = { timestamp: number; value: number; sensor?: string };

export default function DashboardPage() {
  const [data, setData] = useState<Point[]>([]);

  useEffect(() => {
    // Try to fetch latest readings; fallback to mock
    const run = async () => {
      try {
        const r = await fetch("/api/proxy/api/iot/data?limit=50");
        if (r.ok) {
          const arr = await r.json();
          const mapped = Array.isArray(arr) ? arr : (arr.items ?? []);
          setData(mapped.map((it: any, i: number) => ({
            timestamp: Date.parse(it.timestamp ?? it.time ?? new Date().toISOString()) || Date.now() + i * 1000,
            value: Number(it.value ?? it.v ?? Math.random()*100),
            sensor: String(it.sensor ?? "s1")
          })));
          return;
        }
      } catch {}
      // mock
      const now = Date.now();
      setData(Array.from({ length: 50 }).map((_, i) => ({
        timestamp: now - (50 - i) * 1000,
        value: Math.round(40 + 10 * Math.sin(i / 4) + Math.random() * 8),
        sensor: "mock"
      })));
    };
    run();
  }, []);

  return (
    <>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <TenantSwitcher />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ChartCard title="Latest Sensor Values" data={data} />
        <div className="rounded-2xl border p-4 bg-white">
          <div className="font-semibold mb-2">Admin Tools</div>
          <RoleGuard allow={["platform_admin", "client_admin"]}>
            <ul className="list-disc pl-6 text-sm">
              <li>Manage tenants</li>
              <li>View usage and rate limits</li>
              <li>Access audit logs</li>
            </ul>
          </RoleGuard>
          <RoleGuard allow={["operator"]}>
            <div className="text-sm text-gray-600">Operators can monitor live sensor status and acknowledge alerts.</div>
          </RoleGuard>
        </div>
      </div>
    </>
  );
}
