"use client";

import { ReactNode, useEffect, useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import Loading from "./Loading";

export default function Protected({ children }: { children: ReactNode }) {
  const { token, payload, refresh } = useAuth();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const ensure = async () => {
      // Always attempt a refresh-on-load to pick up cookie token
      const ok = await refresh();
      setReady(true);
    };
    ensure();
  }, [refresh]);

  if (!ready) return <Loading />;
  if (!token) {
    if (typeof window !== "undefined") window.location.href = "/login";
    return <Loading />;
  }
  return <>{children}</>;
}
