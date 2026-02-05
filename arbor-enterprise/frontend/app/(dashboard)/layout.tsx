"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Search,
  LayoutDashboard,
  Network,
  Settings,
  Database,
  Terminal,
  Activity,
  Sparkles,
  Upload,
  ChevronDown,
  Bell,
  Plus,
} from "lucide-react";
import { cn } from "@/lib/utils";

const sidebarSections = [
  {
    label: "Platform",
    links: [
      { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
      { href: "/discover", label: "Discover", icon: Search },
      { href: "/map", label: "Knowledge Graph", icon: Network },
    ],
  },
  {
    label: "Management",
    links: [
      { href: "/curator", label: "Curator", icon: Sparkles },
      { href: "/entities", label: "Entities", icon: Database },
      { href: "/ingestion", label: "Ingestion", icon: Upload },
      { href: "/analytics", label: "Analytics", icon: Activity },
    ],
  },
  {
    label: "Account",
    links: [{ href: "/profile", label: "Settings", icon: Settings }],
  },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen bg-gray-50 text-foreground font-sans">
      {/* ─── Sidebar ─── */}
      <aside className="sticky top-0 flex h-screen w-[250px] shrink-0 flex-col border-r border-gray-200 bg-white">
        {/* Workspace Selector */}
        <div className="flex h-14 items-center gap-3 border-b border-gray-200 px-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[#4353FF] to-[#7C3AED] text-white text-sm font-bold">
            A
          </div>
          <div className="flex flex-1 items-center justify-between">
            <div>
              <div className="text-sm font-semibold text-gray-900">ARBOR</div>
              <div className="text-xs text-gray-500">Enterprise</div>
            </div>
            <ChevronDown className="h-4 w-4 text-gray-400" />
          </div>
        </div>

        {/* New Entity Button */}
        <div className="px-3 pt-3 pb-1">
          <Link
            href="/ingestion"
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 transition-colors"
          >
            <Plus className="h-4 w-4" />
            New Entity
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-3 py-2 scrollbar-thin">
          {sidebarSections.map((section) => (
            <div key={section.label} className="mb-4">
              <div className="mb-1 px-3 py-1 text-xs font-medium uppercase tracking-wider text-gray-400">
                {section.label}
              </div>
              <div className="space-y-0.5">
                {section.links.map((link) => {
                  const isActive =
                    pathname === link.href ||
                    (link.href !== "/dashboard" &&
                      pathname.startsWith(link.href));
                  return (
                    <Link
                      key={link.href}
                      href={link.href}
                      className={cn(
                        "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                        isActive
                          ? "bg-blue-50 text-[#4353FF] font-medium"
                          : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                      )}
                    >
                      <link.icon
                        className={cn(
                          "h-4 w-4 shrink-0",
                          isActive
                            ? "text-[#4353FF]"
                            : "text-gray-400 group-hover:text-gray-600"
                        )}
                      />
                      {link.label}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Bottom user section */}
        <div className="border-t border-gray-200 p-3">
          <Link
            href="/profile"
            className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-gray-100 transition-colors"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-500 text-xs font-medium text-white">
              U
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-900 truncate">
                User
              </div>
              <div className="text-xs text-gray-500 truncate">
                user@arbor.ai
              </div>
            </div>
          </Link>
        </div>
      </aside>

      {/* ─── Main content ─── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-gray-200 bg-white px-6">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Link href="/dashboard" className="hover:text-gray-700 transition-colors">
              Dashboard
            </Link>
            {pathname !== "/dashboard" && (
              <>
                <span className="text-gray-300">/</span>
                <span className="text-gray-900 font-medium capitalize">
                  {pathname.split("/").filter(Boolean).pop()?.replace(/-/g, " ")}
                </span>
              </>
            )}
          </div>
          <div className="flex items-center gap-4">
            <button className="relative rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors">
              <Bell className="h-4 w-4" />
            </button>
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-500 text-xs font-medium text-white cursor-pointer">
              U
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-6 py-6 animate-fade-in">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
}
