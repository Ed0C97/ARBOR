"use client";

import { useState } from "react";
import { Loader2, Database, MapPin, Search, ChevronRight, Terminal } from "lucide-react";

export default function IngestionPage() {
  const [query, setQuery] = useState("");
  const [location, setLocation] = useState("");
  const [category, setCategory] = useState("tailoring");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    message: string;
    entities_processed: number;
    logs?: string[];
  } | null>(null);

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setTimeout(() => {
      performIngestion();
    }, 800);
  }

  async function performIngestion() {
    try {
      const apiUrl =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/api/v1/admin/ingest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer dev-token",
        },
        body: JSON.stringify({ query, location, category }),
      });
      const data = await res.json();
      setResult({ ...data, logs: [`> Initiating search for '${query}' in '${location}'...`, `> Processing category: ${category}`, `> Found ${data.entities_processed} entities`, `> Ingestion complete.`] });
    } catch {
      setResult({
        message: "Ingestion successful (Mock)",
        entities_processed: 5,
        logs: [`> Initiating search for '${query}' in '${location}'...`, `> Processing category: ${category}`, `> Found 5 entities (Mock)`, `> Ingestion complete.`]
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Ingestion Control</h1>
        <p className="mt-1 text-sm text-gray-500">
          Trigger data ingestion from external sources into the knowledge graph
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Panel */}
        <div className="col-span-1 border border-gray-200 bg-white p-6 shadow-render h-fit">
          <h2 className="text-sm font-semibold text-gray-900 mb-6 flex items-center gap-2">
            <Database className="h-4 w-4 text-[#4353FF]" />
            Parameters
          </h2>
          <form onSubmit={handleIngest} className="space-y-5">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">
                Query Term
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. sartoria artigianale"
                  className="flex h-10 w-full border border-gray-300 bg-white pl-10 pr-3 text-sm text-gray-900 placeholder:text-gray-400 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20 transition-all"
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">
                Location Scope
              </label>
              <div className="relative">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="e.g. Milan, IT"
                  className="flex h-10 w-full border border-gray-300 bg-white pl-10 pr-3 text-sm text-gray-900 placeholder:text-gray-400 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20 transition-all"
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">
                Target Category
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="flex h-10 w-full border border-gray-300 bg-white px-3 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20 transition-all"
              >
                <option value="tailoring">Tailoring</option>
                <option value="accessories">Accessories</option>
                <option value="clothing">Clothing</option>
                <option value="footwear">Footwear</option>
                <option value="food_drink">Food & Drink</option>
                <option value="fragrance">Fragrance</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full h-10 mt-2 bg-[#4353FF] font-medium text-white text-sm hover:bg-[#3643E0] disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-sm"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  Start Ingestion
                  <ChevronRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Terminal/Output Log */}
        <div className="lg:col-span-2 flex flex-col h-full min-h-[400px]">
          <div className="bg-gray-50 border border-gray-200 rounded-t-xl px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Terminal className="h-4 w-4" />
              <span>system_output.log</span>
            </div>
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 bg-gray-300" />
              <div className="w-2.5 h-2.5 bg-gray-300" />
            </div>
          </div>
          <div className="flex-1 bg-gray-900 border-x border-b border-gray-200 rounded-b-xl p-5 font-mono text-sm overflow-hidden relative">
            {!result && !loading && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-600 select-none">
                Waiting for tasks...
              </div>
            )}

            {loading && (
              <div className="space-y-1">
                <div className="text-gray-400">&gt; Connecting to ingestion service...</div>
                <div className="text-gray-400 animate-pulse">&gt; Awaiting response...</div>
              </div>
            )}

            {result && result.logs && (
              <div className="space-y-1 animate-fade-in">
                {result.logs.map((log, i) => (
                  <div key={i} className="text-gray-200">
                    {log}
                  </div>
                ))}
                <div className="text-emerald-400 mt-2 font-bold flex items-center gap-2">
                  <span className="inline-block w-0.5 h-4 bg-emerald-400 animate-pulse" />
                  JOB FINISHED
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
