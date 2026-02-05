"use client";

import Link from "next/link";
import { MapPin, Tag, ArrowUpRight } from "lucide-react";
import type { RecommendationItem } from "@/lib/types";

interface RecommendationCardProps {
  recommendation: RecommendationItem;
}

export function RecommendationCard({
  recommendation,
}: RecommendationCardProps) {
  const scorePercent = Math.round(recommendation.score * 100);

  return (
    <Link href={`/entity/${recommendation.id}`} className="block h-full">
      <div className="group relative h-full flex flex-col justify-between border border-gray-200 bg-white p-4 transition-all hover:border-[#4353FF]/30 hover:shadow-render-md">
        <div className="space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-900 group-hover:text-[#4353FF] transition-colors text-sm">
                {recommendation.name}
              </h4>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                {recommendation.category && (
                  <span className="flex items-center gap-1">
                    <Tag className="h-3 w-3" /> {recommendation.category}
                  </span>
                )}
                {recommendation.city && (
                  <span className="flex items-center gap-1">
                    <MapPin className="h-3 w-3" /> {recommendation.city}
                  </span>
                )}
              </div>
            </div>
            <div className="flex flex-col items-end gap-0.5 shrink-0">
              <span className="text-lg font-bold text-[#4353FF] leading-none">
                {scorePercent}%
              </span>
              <span className="text-[10px] text-gray-400 uppercase">
                Match
              </span>
            </div>
          </div>
          {recommendation.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {recommendation.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center rounded-md bg-gray-100 px-2 py-0.5 text-[10px] text-gray-600"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="mt-4 pt-3 border-t border-gray-100 flex items-center justify-between text-xs text-gray-400 group-hover:text-[#4353FF] transition-colors">
          <span>View Entity</span>
          <ArrowUpRight className="h-3 w-3" />
        </div>
      </div>
    </Link>
  );
}
