"use client";

import Protected from "@/components/Protected";
import DataTable from "@/components/DataTable";
import { useEffect, useState } from "react";
import { formatDate } from "@/lib/utils";

type Row = { id: string; ts: string | number; actor: string; action: string; target?: string; details?: string };

export default function AuditPage() {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    const run = async () => {
      try {
        const r = await fetch("/api/proxy/audit/logs?limit=100");
        if (r.ok) {
          const data = await r.json();
          const items = Array.isArray(data) ? data : (data.items ?? []);
          setRows(items.map((x: any, i: number) => ({
            id: String(x.id ?? i),
            ts: x.ts ?? x.timestamp ?? new Date().toISOString(),
            actor: x.actor ?? x.user ?? "system",
            action: x.action ?? x.event ?? "unknown",
            target: x.target ?? x.resource ?? "",
            details: typeof x.details === "string" ? x.details : JSON.stringify(x.details ?? {}),
          })));
          return;
        }
      } catch {}
      setRows([]);
    };
    run();
  }, []);

  return (
    <Protected>
      <h1 className="text-2xl font-semibold mb-4">Audit Log</h1>
      <DataTable
        columns={[
          { key: "ts", header: "Time", render: (v) => formatDate(v) },
          { key: "actor", header: "Actor" },
          { key: "action", header: "Action" },
          { key: "target", header: "Target" },
          { key: "details", header: "Details" },
        ]}
        rows={rows}
      />
    </Protected>
  );
}
