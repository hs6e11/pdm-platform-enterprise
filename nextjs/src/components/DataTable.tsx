type Column<T> = { key: keyof T; header: string; render?: (v: any, row: T) => any };

export default function DataTable<T extends { [key: string]: any }>({ columns, rows }: { columns: Column<T>[]; rows: T[]; }) {
  return (
    <div className="rounded-2xl border bg-white overflow-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((c, i) => (
              <th key={i} className="text-left font-semibold px-3 py-2 border-b">{c.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="odd:bg-white even:bg-gray-50">
              {columns.map((c, j) => (
                <td key={j} className="px-3 py-2 border-b">{c.render ? c.render(r[c.key], r) : String(r[c.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
