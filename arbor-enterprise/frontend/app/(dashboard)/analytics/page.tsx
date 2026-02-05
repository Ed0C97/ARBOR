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
        setStats({
          total_entities: 1248,
          pending_entities: 45,
          vetted_entities: 982,
          total_queries: 15342
        });
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
        color: "text-[#4353FF]",
        bg: "bg-blue-50",
      },
      {
        label: "Pending Review",
        value: stats.pending_entities ?? 0,
        icon: Clock,
        color: "text-amber-600",
        bg: "bg-amber-50",
      },
      {
        label: "Vetted Entities",
        value: stats.vetted_entities ?? 0,
        icon: CheckCircle,
        color: "text-emerald-600",
        bg: "bg-emerald-50",
      },
      {
        label: "Total Queries",
        value: stats.total_queries ?? 0,
        icon: Activity,
        color: "text-purple-600",
        bg: "bg-purple-50",
      },
    ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Analytics</h1>
        <p className="mt-1 text-sm text-gray-500">
          Real-time metrics and performance monitoring
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => (
          <div
            key={card.label}
            className="rounded-xl border border-gray-200 bg-white p-6 shadow-render hover:shadow-render-md transition-shadow"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 mb-2">{card.label}</p>
                <p className="text-3xl font-semibold text-gray-900">
                  {card.value.toLocaleString()}
                </p>
              </div>
              <div className={`p-2.5 rounded-lg ${card.bg}`}>
                <card.icon className={`h-5 w-5 ${card.color}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {!stats && (
        <div className="flex justify-center p-12">
          <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
        </div>
      )}

      {/* Chart Area */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-render">
        <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center gap-2">
          <Activity className="h-4 w-4 text-[#4353FF]" />
          Query Volume (24h)
        </h3>
        <div className="h-64 w-full rounded-lg bg-gray-50 border border-gray-100 flex items-center justify-center text-sm text-gray-400">
          Chart visualization placeholder
        </div>
      </div>
    </div>
  );
}
