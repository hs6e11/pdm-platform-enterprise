"use client";

import { ReactNode } from "react";
import { useAuth } from "@/hooks/useAuth";
import type { Role } from "@/lib/roles";

export default function RoleGuard({ allow, children }: { allow: Role[]; children: ReactNode; }) {
  const { role } = useAuth();
  if (!role) return null;
  if (!allow.includes(role as Role)) return <div className="p-6 text-red-600">You do not have access to this section.</div>;
  return <>{children}</>;
}
