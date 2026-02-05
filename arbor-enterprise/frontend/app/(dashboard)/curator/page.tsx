"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import {
  Sparkles,
  Tag as TagIcon,
  Loader2,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Filter
} from "lucide-react";
import type { Entity, EntityFilters, EntityListResponse, EntityType } from "@/lib/types";
import { listEntities, enrichEntity } from "@/lib/api";

export default function CuratorPage() {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [enriching, setEnriching] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [limit] = useState(20);
  const [filter, setFilter] = useState<{
    entity_type: EntityType | "";
    category: string;
    is_featured: string;
  }>({ entity_type: "", category: "", is_featured: "" });

  const fetchEntities = useCallback(async () => {
    setLoading(true);
    try {
      const filters: EntityFilters = {
        offset: page * limit,
        limit,
      };
      if (filter.entity_type) filters.entity_type = filter.entity_type;
      if (filter.category) filters.category = filter.category;
      if (filter.is_featured === "true") filters.is_featured = true;
      if (filter.is_featured === "false") filters.is_featured = false;

      const data: EntityListResponse = await listEntities(filters);
      setEntities(data.items || []);
      setTotal(data.total);
    } catch {
      console.error("Failed to fetch entities");
      setEntities([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, limit, filter]);

  useEffect(() => {
    fetchEntities();
  }, [fetchEntities]);

  useEffect(() => {
    setPage(0);
  }, [filter]);

  const totalPages = Math.ceil(total / limit);
  const rangeStart = total === 0 ? 0 : page * limit + 1;
  const rangeEnd = Math.min((page + 1) * limit, total);

  async function handleEnrich(entityId: string) {
    setEnriching(entityId);
    try {
      await enrichEntity(entityId, {
        tags: ["curated", "arbor-reviewed"],
        vibe_dna: {
          dimensions: {
            formality: 0.7,
            craftsmanship: 0.8,
            price_value: 0.6,
            atmosphere: 0.7,
            service_quality: 0.8,
            exclusivity: 0.5,
          },
        },
      });
      fetchEntities();
    } catch {
      console.error("Failed to enrich entity");
    } finally {
      setEnriching(null);
    }
  }

  const typeColors: Record<string, string> = {
    brand: "text-amber-700 border-amber-200 bg-amber-50",
    venue: "text-blue-700 border-blue-200 bg-blue-50",
  };

  const pageNumbers = useMemo(() => {
    const pages: (number | "ellipsis-start" | "ellipsis-end")[] = [];
    if (totalPages <= 7) {
      for (let i = 0; i < totalPages; i++) pages.push(i);
    } else {
      pages.push(0);
      if (page > 3) pages.push("ellipsis-start");
      const start = Math.max(1, page - 2);
      const end = Math.min(totalPages - 2, page + 2);
      for (let i = start; i <= end; i++) pages.push(i);
      if (page < totalPages - 4) pages.push("ellipsis-end");
      pages.push(totalPages - 1);
    }
    return pages;
  }, [page, totalPages]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Curator Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Browse, review, and enrich entities in the knowledge graph
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center border border-gray-200 bg-white p-4 shadow-render">
        <Filter className="h-4 w-4 text-gray-400 mr-1" />
        <select
          value={filter.entity_type}
          onChange={(e) =>
            setFilter({ ...filter, entity_type: e.target.value as EntityType | "" })
          }
          className="h-9 border border-gray-300 bg-white px-3 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20"
        >
          <option value="">All types</option>
          <option value="brand">Brand</option>
          <option value="venue">Venue</option>
        </select>
        <select
          value={filter.category}
          onChange={(e) => setFilter({ ...filter, category: e.target.value })}
          className="h-9 border border-gray-300 bg-white px-3 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20"
        >
          <option value="">All categories</option>
          <option value="tailoring">Tailoring</option>
          <option value="accessories">Accessories</option>
          <option value="clothing">Clothing</option>
          <option value="footwear">Footwear</option>
          <option value="jewelry">Jewelry</option>
          <option value="food_drink">Food &amp; Drink</option>
          <option value="restaurant">Restaurant</option>
          <option value="bar">Bar</option>
          <option value="hotel">Hotel</option>
        </select>
        <select
          value={filter.is_featured}
          onChange={(e) =>
            setFilter({ ...filter, is_featured: e.target.value })
          }
          className="h-9 border border-gray-300 bg-white px-3 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20"
        >
          <option value="">All entities</option>
          <option value="true">Featured</option>
          <option value="false">Standard</option>
        </select>
      </div>

      {/* Entity table */}
      <div className="rounded border border-gray-200 bg-white overflow-hidden shadow-render">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-gray-200 bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Name</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Type</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Category</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">City</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Price</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Enrichment</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center">
                  <Loader2 className="mx-auto h-5 w-5 animate-spin text-gray-400" />
                </td>
              </tr>
            ) : entities.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center text-sm text-gray-400">
                  No entities found matching criteria
                </td>
              </tr>
            ) : (
              entities.map((entity) => (
                <tr key={entity.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-900 text-sm">
                      {entity.name}
                    </div>
                    {entity.instagram && (
                      <div className="text-xs text-gray-400 mt-0.5">
                        @{entity.instagram.replace("@", "")}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 text-xs font-medium capitalize rounded-md border ${typeColors[entity.entity_type] || "border-gray-200 text-gray-500 bg-gray-50"}`}>
                      {entity.entity_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 capitalize text-gray-600 text-sm">
                    {entity.category}
                  </td>
                  <td className="px-4 py-3 text-gray-600 text-sm">
                    {entity.city || "\u2014"}
                  </td>
                  <td className="px-4 py-3 text-gray-600 text-sm">
                    {entity.price_range !== null ? "$".repeat(Number(entity.price_range)) : "\u2014"}
                  </td>
                  <td className="px-4 py-3">
                    {entity.vibe_dna ? (
                      <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-600">
                        <Sparkles className="h-3 w-3" />
                        Enriched
                      </span>
                    ) : entity.tags && entity.tags.length > 0 ? (
                      <span className="flex items-center gap-1.5 text-xs font-medium text-amber-600">
                        <TagIcon className="h-3 w-3" />
                        Tagged
                      </span>
                    ) : (
                      <span className="text-xs text-gray-400">
                        Pending
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-2">
                      {!entity.vibe_dna && (
                        <button
                          onClick={() => handleEnrich(entity.id)}
                          disabled={enriching === entity.id}
                          className="flex items-center gap-1.5 bg-[#4353FF] text-white px-3 py-1.5 text-xs font-medium hover:bg-[#3643E0] disabled:opacity-50 transition-colors"
                        >
                          {enriching === entity.id ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            <Sparkles className="h-3 w-3" />
                          )}
                          Enrich
                        </button>
                      )}
                      <a
                        href={`/entity/${entity.id}`}
                        className="border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 hover:border-gray-300 transition-colors"
                      >
                        View
                      </a>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 0 && (
        <div className="flex flex-col items-center justify-between gap-4 sm:flex-row pt-2">
          <span className="text-sm text-gray-500">
            Showing <span className="font-medium text-gray-900">{rangeStart}&ndash;{rangeEnd}</span> of {total}
          </span>

          <div className="flex items-center gap-1">
            <button
              disabled={page === 0}
              onClick={() => setPage(0)}
              className="flex h-8 w-8 items-center justify-center text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:opacity-30 transition-colors"
            >
              <ChevronsLeft className="h-4 w-4" />
            </button>
            <button
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
              className="flex h-8 w-8 items-center justify-center text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:opacity-30 transition-colors"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>

            {pageNumbers.map((p, idx) =>
              typeof p === "string" ? (
                <span
                  key={p + idx}
                  className="flex h-8 w-8 items-center justify-center text-sm text-gray-400"
                >
                  &hellip;
                </span>
              ) : (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`flex h-8 w-8 items-center justify-center text-sm transition-colors ${
                    p === page
                      ? "bg-[#4353FF] text-white"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  {p + 1}
                </button>
              ),
            )}

            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(page + 1)}
              className="flex h-8 w-8 items-center justify-center text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:opacity-30 transition-colors"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(totalPages - 1)}
              className="flex h-8 w-8 items-center justify-center text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:opacity-30 transition-colors"
            >
              <ChevronsRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
