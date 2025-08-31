"use client";

import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";

export default function LoginPage() {
  const { login, loading } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [tenant, setTenant] = useState("");

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await login(username, password, tenant || undefined);
  };

  return (
    <div className="mx-auto max-w-md mt-16">
      <h1 className="text-2xl font-semibold mb-6">Sign in</h1>
      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label className="block text-sm mb-1">Username</label>
          <input className="w-full border rounded-md px-3 py-2" value={username} onChange={e => setUsername(e.target.value)} />
        </div>
        <div>
          <label className="block text-sm mb-1">Password</label>
          <input className="w-full border rounded-md px-3 py-2" type="password" value={password} onChange={e => setPassword(e.target.value)} />
        </div>
        <div>
          <label className="block text-sm mb-1">Tenant (optional)</label>
          <input className="w-full border rounded-md px-3 py-2" value={tenant} onChange={e => setTenant(e.target.value)} placeholder="tenant id" />
        </div>
        <button disabled={loading} className="w-full rounded-md bg-gray-900 text-white px-3 py-2">
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}
