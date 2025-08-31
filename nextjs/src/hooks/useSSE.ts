"use client";

import { useEffect, useRef, useState } from "react";

type DataPoint = { timestamp: number; value: number; sensor?: string };

export function useSSE(url?: string) {
  const [data, setData] = useState<DataPoint[]>([]);
  const evtSource = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!url) return;
    const es = new EventSource(url, { withCredentials: true });
    evtSource.current = es;
    es.onmessage = (e) => {
      try {
        const obj = JSON.parse(e.data);
        setData((prev) => [...prev.slice(-499), obj]);
      } catch {}
    };
    es.onerror = () => {
      es.close();
    };
    return () => es.close();
  }, [url]);

  return data;
}
