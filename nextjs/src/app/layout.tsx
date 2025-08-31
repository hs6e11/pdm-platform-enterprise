import "./globals.css";
import NavBar from "@/components/NavBar";

export const metadata = {
  title: "PDM Frontend",
  description: "Next.js + Tailwind frontend for PdM platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <NavBar />
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
