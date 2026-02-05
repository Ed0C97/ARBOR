"use client";

import { useEffect, useState } from "react";
import { Loader2, Activity, Database, CheckCircle, Clock } from "lucide-react";

interface Stats {
  total_entities: number;
  pending_entities: number;
  vetted_entities: number;
  total_queries: number;
}

export default function AnalyticsPage() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    async function fetchStats() {
      try {
        const apiUrl =
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/api/v1/admin/stats`, {
          headers: { Authorization: "Bearer dev-token" },
        });
        const data = await res.json();
        setStats(data);
      } catch {
        // Fallback for demo/dev if API fails
        setStats({
          total_entities: 1248,
          pending_entities: 45,
          vetted_entities: 982,
          total_queries: 15342
        })
      }
    }
    fetchStats();
  }, []);

  const cards = stats
    ? [
      {
        label: "Total Entities",
        value: stats.total_entities ?? 0,
        icon: Database,
        color: "text-jetbrains-blue",
        border: "border-l-jetbrains-blue",
      },
      {
        label: "Pending Review",
        value: stats.pending_entities ?? 0,
        icon: Clock,
        color: "text-jetbrains-orange",
        border: "border-l-jetbrains-orange",
      },
      {
        label: "Vetted Entities",
        value: stats.vetted_entities ?? 0,
        icon: CheckCircle,
        color: "text-jetbrains-green",
        border: "border-l-jetbrains-green",
      },
      {
        label: "Total Queries",
        value: stats.total_queries ?? 0,
        icon: Activity,
        color: "text-jetbrains-purple",
        border: "border-l-jetbrains-purple",
      },
    ]
    : [];

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-jetbrains-contrast">System Analytics</h1>
        <p className="text-sm text-jetbrains-gray-light font-mono">
          &gt; Real-time metrics and performance monitoring
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => (
          <div
            key={card.label}
            className={`group relative overflow-hidden rounded-sm border border-jetbrains-gray/30 bg-jetbrains-dark p-6 shadow-sm transition-all hover:shadow-glow hover:border-jetbrains-gray/50`}
          >
            <div className={`absolute left-0 top-0 bottom-0 w-1 ${card.border}`} />

            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-mono font-medium text-jetbrains-gray-light uppercase tracking-wider mb-2">{card.label}</p>
                <p className="text-3xl font-mono font-bold text-jetbrains-contrast">
                  {card.value.toLocaleString()}
                </p>
              </div>
              <div className={`p-2 rounded-sm bg-jetbrains-ink ${card.color}`}>
                <card.icon className="h-5 w-5" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {!stats && (
        <div className="flex justify-center p-12">
          <Loader2 className="h-8 w-8 animate-spin text-jetbrains-blue" />
        </div>
      )}

      {/* Mock Chart Area */}
      <div className="rounded-sm border border-jetbrains-gray/30 bg-jetbrains-dark p-6">
        <h3 className="text-sm font-medium text-jetbrains-contrast mb-4 flex items-center gap-2">
          <Activity className="h-4 w-4 text-jetbrains-blue" />
          Query Volume (24h)
        </h3>
        <div className="h-64 w-full bg-jetbrains-ink/50 border border-jetbrains-gray/10 flex items-center justify-center text-xs font-mono text-jetbrains-gray-light">
          [CHART VISUALIZATION PLACEHOLDER]
        </div>
      </div>
    </div>
  );
}
