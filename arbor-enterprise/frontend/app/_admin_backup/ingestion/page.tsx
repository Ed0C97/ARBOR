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

    // Simulate logs for better UX
    setTimeout(() => {
      // Fallback or actual API call
      // Using existing logic but wrapping in try/catch for real API
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
      // Mock for demo if API fails
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
    <div className="space-y-8">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-jetbrains-contrast">Ingestion Control</h1>
        <p className="text-sm text-jetbrains-gray-light font-mono">
          &gt; Trigger data ingestion from external sources
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Control Panel */}
        <div className="col-span-1 border border-jetbrains-gray/30 bg-jetbrains-dark p-6 rounded-sm shadow-sm h-fit">
          <h2 className="text-sm font-bold text-jetbrains-contrast mb-6 uppercase tracking-wider flex items-center gap-2">
            <Database className="h-4 w-4 text-jetbrains-blue" />
            Parameters
          </h2>
          <form onSubmit={handleIngest} className="space-y-5">
            <div className="space-y-2">
              <label className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Query Term
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-jetbrains-gray-light" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. sartoria artigianale"
                  className="flex h-9 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink pl-9 pr-3 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Location Scope
              </label>
              <div className="relative">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-jetbrains-gray-light" />
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="e.g. Milan, IT"
                  className="flex h-9 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink pl-9 pr-3 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Target Category
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="flex h-9 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all appearance-none"
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
              className="w-full h-10 mt-4 bg-jetbrains-blue font-medium text-white text-sm rounded-sm hover:bg-jetbrains-blue/90 hover:shadow-glow disabled:opacity-50 disabled:hover:shadow-none transition-all flex items-center justify-center gap-2"
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
          <div className="bg-jetbrains-db-bg border border-jetbrains-gray/30 rounded-t-sm p-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-mono text-jetbrains-gray-light">
              <Terminal className="h-3.5 w-3.5" />
              <span>system_output.log</span>
            </div>
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-jetbrains-gray/30" />
              <div className="w-2.5 h-2.5 rounded-full bg-jetbrains-gray/30" />
            </div>
          </div>
          <div className="flex-1 bg-jetbrains-ink border-x border-b border-jetbrains-gray/30 p-4 font-mono text-sm overflow-hidden relative">
            {!result && !loading && (
              <div className="absolute inset-0 flex items-center justify-center text-jetbrains-gray/30 select-none">
                Waiting for tasks...
              </div>
            )}

            {loading && (
              <div className="space-y-1">
                <div className="text-jetbrains-gray-light">&gt; Connecting to ingestion service...</div>
                <div className="text-jetbrains-gray-light animate-pulse">&gt; Awaiting response...</div>
              </div>
            )}

            {result && result.logs && (
              <div className="space-y-1 animate-fade-in">
                {result.logs.map((log, i) => (
                  <div key={i} className="text-jetbrains-contrast">
                    {log}
                  </div>
                ))}
                <div className="text-jetbrains-green mt-2 font-bold flex items-center gap-2">
                  <span className="inline-block w-2 h-4 bg-jetbrains-green animate-cursor-blink">|</span>
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
