export type Role = "platform_admin" | "client_admin" | "operator";

export const ROLE_LABELS: Record<Role, string> = {
  platform_admin: "Platform Admin",
  client_admin: "Client Admin",
  operator: "Operator",
};
