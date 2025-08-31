export function formatDate(d: string | number | Date) {
  const date = d instanceof Date ? d : new Date(d);
  return date.toLocaleString();
}
