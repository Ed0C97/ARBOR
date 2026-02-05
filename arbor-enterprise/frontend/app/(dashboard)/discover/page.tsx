"use client";

import { useState } from "react";
import { Search, Sparkles } from "lucide-react";
import { ChatInterface } from "@/components/chat/ChatInterface";

export default function DiscoverPage() {
  const [hasStarted, setHasStarted] = useState(false);

  return (
    <div className="flex flex-col h-[calc(100vh-7rem)]">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Discovery Engine</h1>
        <p className="mt-1 text-sm text-gray-500">
          Query the knowledge graph using natural language
        </p>
      </div>

      {/* Empty state */}
      {!hasStarted ? (
        <div className="flex-1 flex flex-col items-center justify-center rounded-xl border border-gray-200 bg-white p-10 shadow-render">
          <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
            <Search className="h-7 w-7" />
          </div>
          <div className="max-w-lg text-center space-y-3 mb-10">
            <h2 className="text-lg font-semibold text-gray-900">
              Start a discovery session
            </h2>
            <p className="text-sm text-gray-500 leading-relaxed">
              Enter a prompt to search across entities, vibes, and
              relationships. Use natural language to describe what you are
              looking for.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
            {[
              { label: "Find classic wine bars in Rome", sub: "Location + category" },
              { label: "High-end minimal menswear", sub: "Style + category" },
              { label: "Hidden speakeasy vibes", sub: "Atmosphere search" },
              { label: "Avant-garde art galleries", sub: "Niche discovery" },
            ].map((item, i) => (
              <button
                key={i}
                onClick={() => setHasStarted(true)}
                className="group flex flex-col items-start gap-1 rounded-lg border border-gray-200 bg-white p-4 text-left transition-all hover:border-[#4353FF]/30 hover:bg-blue-50/50 hover:shadow-sm"
              >
                <span className="text-sm font-medium text-gray-900 group-hover:text-[#4353FF] transition-colors">
                  {item.label}
                </span>
                <span className="text-xs text-gray-400">{item.sub}</span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 min-h-0 rounded-xl border border-gray-200 bg-white overflow-hidden shadow-render">
          <ChatInterface onFirstMessage={() => {}} />
        </div>
      )}
    </div>
  );
}
