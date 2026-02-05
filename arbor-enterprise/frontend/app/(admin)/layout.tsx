"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Microscope,
  Database,
  BarChart3,
  Upload,
  ArrowLeft,
  Terminal,
} from "lucide-react";
import { cn } from "@/lib/utils";

const adminNav = [
  { href: "/curator", label: "Curator", icon: Microscope },
  { href: "/entities", label: "Entities", icon: Database },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/ingestion", label: "Ingestion", icon: Upload },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="flex h-screen bg-jetbrains-ink text-jetbrains-contrast font-sans">
      {/* Admin Sidebar */}
      <aside className="w-64 border-r border-jetbrains-gray/30 bg-jetbrains-dark flex flex-col">
        <div className="p-4 mb-4">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-8 w-8 rounded-sm bg-gradient-to-br from-jetbrains-blue to-jetbrains-purple flex items-center justify-center text-white shadow-glow">
              <span className="font-mono font-bold text-lg">A</span>
            </div>
            <h1 className="font-bold tracking-tight text-jetbrains-contrast text-lg">
              ARBOR<span className="text-jetbrains-blue">.SYS</span>
            </h1>
          </div>
          <div className="px-1">
            <div className="text-[10px] font-mono text-jetbrains-gray-light uppercase tracking-wider bg-jetbrains-ink border border-jetbrains-gray/30 py-1 px-2 rounded-sm inline-block">
              Admin Console
            </div>
          </div>
        </div>

        <nav className="flex-1 px-2 space-y-1">
          {adminNav.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 text-sm transition-all duration-200 rounded-sm group font-mono",
                  isActive
                    ? "bg-jetbrains-blue/10 text-jetbrains-contrast font-medium shadow-sm border border-jetbrains-blue/20"
                    : "text-jetbrains-gray-light hover:bg-jetbrains-ink hover:text-jetbrains-contrast border border-transparent"
                )}
              >
                <item.icon className={cn("h-4 w-4 transition-colors", isActive ? "text-jetbrains-blue" : "text-jetbrains-gray group-hover:text-jetbrains-contrast")} />
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-jetbrains-gray/20">
          <Link
            href="/discover"
            className="flex items-center gap-2 text-xs font-mono text-jetbrains-gray-light transition-colors hover:text-jetbrains-contrast group"
          >
            <ArrowLeft className="h-3 w-3 group-hover:-translate-x-0.5 transition-transform" />
            Back to Discovery
          </Link>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-jetbrains-ink">
        <div className="p-8 max-w-7xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}
