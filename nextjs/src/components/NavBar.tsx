"use client";
import Link from "next/link";
import { useAuth } from "@/hooks/useAuth";
import { useTenant } from "@/hooks/useTenant";

export default function NavBar() {
  const { logout, role } = useAuth();
  const { current } = useTenant();

  return (
    <div className="w-full border-b bg-white">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/dashboard" className="font-semibold">PDM Frontend</Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link href="/dashboard">Dashboard</Link>
            <Link href="/live">Live Charts</Link>
            <Link href="/audit">Audit</Link>
            <Link href="/tenants">Tenants</Link>
          </nav>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="hidden sm:inline text-gray-500">Role: {role ?? "?"}</span>
          <span className="hidden sm:inline text-gray-500">Tenant: {current ?? "-"}</span>
          <button onClick={logout} className="rounded-md bg-gray-900 text-white px-3 py-1.5">Logout</button>
        </div>
      </div>
    </div>
  );
}
