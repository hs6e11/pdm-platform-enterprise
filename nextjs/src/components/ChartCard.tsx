"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

type Point = { timestamp: number; value: number; sensor?: string };

export default function ChartCard({ title, data }: { title: string; data: Point[] }) {
  const chartData = data.map(d => ({ time: new Date(d.timestamp).toLocaleTimeString(), value: d.value }));
  return (
    <div className="rounded-2xl border p-4 shadow-sm bg-white">
      <div className="mb-3 font-semibold">{title}</div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="value" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
