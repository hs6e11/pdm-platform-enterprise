"use client";

import Protected from "@/components/Protected";
import dynamic from "next/dynamic";
const ChartCard = dynamic(() => import("@/components/ChartCard"), { ssr: false });
import { useEffect, useMemo, useRef, useState } from "react";


type Point = { timestamp: number; value: number; sensor?: string };

export default function LivePage() {
  const [items, setItems] = useState<Point[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_SENSOR_WS_URL || "";
    const sseUrl = process.env.NEXT_PUBLIC_SENSOR_SSE_URL || process.env.SENSOR_SSE_URL || "";
    let closed = false;

    if (wsUrl) {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        ws.onmessage = (ev) => {
          try {
            const obj = JSON.parse(ev.data);
            setItems((prev) => [...prev.slice(-999), obj]);
          } catch {}
        };
        ws.onclose = () => { if (!closed) setTimeout(() => window.location.reload(), 2000); };
        return () => { closed = true; ws.close(); };
      } catch {}
    }

    if (sseUrl) {
      const es = new EventSource(sseUrl, { withCredentials: true });
      es.onmessage = (e) => {
        try {
          const obj = JSON.parse(e.data);
          setItems((prev) => [...prev.slice(-999), obj]);
        } catch {}
      };
      es.onerror = () => es.close();
      return () => es.close();
    }

    // Fallback: poll /api/iot/data periodically
    let t = setInterval(async () => {
      try {
        const r = await fetch("/api/proxy/api/iot/data?limit=1");
        if (r.ok) {
          const arr = await r.json();
          const it = Array.isArray(arr) ? arr[0] : (arr.items?.[0]);
          const p: Point = {
            timestamp: Date.now(),
            value: Number(it?.value ?? Math.random() * 100),
            sensor: String(it?.sensor ?? "mock")
          };
          setItems((prev) => [...prev.slice(-999), p]);
        }
      } catch {
        const p: Point = { timestamp: Date.now(), value: Math.random() * 100, sensor: "mock" };
        setItems((prev) => [...prev.slice(-999), p]);
      }
    }, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <Protected>
      <h1 className="text-2xl font-semibold mb-6">Live Charts</h1>
      <ChartCard title="Live Sensor Stream" data={items} />
    </Protected>
  );
}
