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

const ROWS_PER_PAGE_OPTIONS = [10, 15, 20, 25, 50];

export default function CuratorPage() {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [enriching, setEnriching] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [limit, setLimit] = useState(20);
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

  /* Reset page when filters change */
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
    brand: "text-jetbrains-orange border-jetbrains-orange/30 bg-jetbrains-orange/10",
    venue: "text-jetbrains-blue border-jetbrains-blue/30 bg-jetbrains-blue/10",
  };

  /* ---- pagination page numbers ---- */
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
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-jetbrains-contrast">Curator Dashboard</h1>
        <p className="text-sm text-jetbrains-gray-light font-mono">
          &gt; Browse and enrich entities
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center bg-jetbrains-dark border border-jetbrains-gray/30 p-4 rounded-sm">
        <Filter className="h-4 w-4 text-jetbrains-gray-light mr-1" />
        <select
          value={filter.entity_type}
          onChange={(e) =>
            setFilter({ ...filter, entity_type: e.target.value as EntityType | "" })
          }
          className="h-8 rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-xs font-mono text-jetbrains-contrast focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue"
        >
          <option value="">[All types]</option>
          <option value="brand">Brand</option>
          <option value="venue">Venue</option>
        </select>
        <select
          value={filter.category}
          onChange={(e) => setFilter({ ...filter, category: e.target.value })}
          className="h-8 rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-xs font-mono text-jetbrains-contrast focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue"
        >
          <option value="">[All categories]</option>
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
          className="h-8 rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-xs font-mono text-jetbrains-contrast focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue"
        >
          <option value="">[All entities]</option>
          <option value="true">Featured</option>
          <option value="false">Standard</option>
        </select>
      </div>

      {/* Entity table */}
      <div className="rounded-sm border border-jetbrains-gray/30 bg-jetbrains-dark overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-jetbrains-gray/30 bg-jetbrains-ink">
            <tr>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">Name</th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">Type</th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">Category</th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">City</th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">Price</th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">Enrichment</th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-jetbrains-gray/10">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center">
                  <Loader2 className="mx-auto h-6 w-6 animate-spin text-jetbrains-blue" />
                </td>
              </tr>
            ) : entities.length === 0 ? (
              <tr>
                <td
                  colSpan={7}
                  className="px-4 py-12 text-center text-jetbrains-gray font-mono text-sm"
                >
                  &lt; No entities found matching criteria &gt;
                </td>
              </tr>
            ) : (
              entities.map((entity) => (
                <tr
                  key={entity.id}
                  className="hover:bg-jetbrains-gray/5 transition-colors group"
                >
                  <td className="px-4 py-3">
                    <div className="font-medium text-jetbrains-contrast font-sans text-sm">
                      {entity.name}
                    </div>
                    {entity.instagram && (
                      <div className="text-xs text-jetbrains-gray-light font-mono mt-0.5">
                        @{entity.instagram.replace("@", "")}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wider border rounded-sm ${typeColors[entity.entity_type] || "border-jetbrains-gray/30 text-jetbrains-gray"
                        }`}
                    >
                      {entity.entity_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 capitalize text-jetbrains-gray-light text-sm font-sans">
                    {entity.category}
                  </td>
                  <td className="px-4 py-3 text-jetbrains-gray-light text-sm">
                    {entity.city || "\u2014"}
                  </td>
                  <td className="px-4 py-3 font-mono text-jetbrains-gray-light text-xs">
                    {entity.price_range !== null ? "$".repeat(Number(entity.price_range)) : "\u2014"}
                  </td>
                  <td className="px-4 py-3">
                    {entity.vibe_dna ? (
                      <span className="flex items-center gap-1.5 text-xs text-jetbrains-green font-mono">
                        <Sparkles className="h-3 w-3" />
                        ENRICHED
                      </span>
                    ) : entity.tags && entity.tags.length > 0 ? (
                      <span className="flex items-center gap-1.5 text-xs text-jetbrains-orange font-mono">
                        <TagIcon className="h-3 w-3" />
                        TAGGED
                      </span>
                    ) : (
                      <span className="text-xs text-jetbrains-gray/50 font-mono">
                        PENDING
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-2">
                      {!entity.vibe_dna && (
                        <button
                          onClick={() => handleEnrich(entity.id)}
                          disabled={enriching === entity.id}
                          className="flex items-center gap-1.5 bg-jetbrains-blue text-white px-2 py-1 text-[10px] rounded-sm font-mono hover:bg-jetbrains-blue/90 disabled:opacity-50 transition-colors uppercase tracking-wider"
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
                        className="bg-jetbrains-ink border border-jetbrains-gray/30 px-2 py-1 text-[10px] font-mono rounded-sm text-jetbrains-gray-light hover:text-jetbrains-contrast hover:border-jetbrains-gray transition-colors uppercase tracking-wider block"
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
        <div className="flex flex-col items-center justify-between gap-4 sm:flex-row border-t border-jetbrains-gray/20 pt-4">
          {/* Left: range indicator */}
          <span className="font-mono text-xs text-jetbrains-gray-light">
            Showing <span className="text-jetbrains-contrast">{rangeStart}&ndash;{rangeEnd}</span> of {total}
          </span>

          {/* Right: page numbers + arrows */}
          <div className="flex items-center gap-1">
            <button
              disabled={page === 0}
              onClick={() => setPage(0)}
              className="flex h-7 w-7 items-center justify-center border border-transparent hover:bg-jetbrains-gray/10 text-jetbrains-gray-light rounded-sm disabled:opacity-30 transition-colors"
            >
              <ChevronsLeft className="h-4 w-4" />
            </button>
            <button
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
              className="flex h-7 w-7 items-center justify-center border border-transparent hover:bg-jetbrains-gray/10 text-jetbrains-gray-light rounded-sm disabled:opacity-30 transition-colors"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>

            {pageNumbers.map((p, idx) =>
              typeof p === "string" ? (
                <span
                  key={p + idx}
                  className="flex h-7 w-7 items-center justify-center font-mono text-xs text-jetbrains-gray"
                >
                  &hellip;
                </span>
              ) : (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`flex h-7 w-7 items-center justify-center font-mono text-xs rounded-sm transition-colors ${p === page
                    ? "bg-jetbrains-blue text-white"
                    : "text-jetbrains-gray-light hover:bg-jetbrains-gray/10 hover:text-jetbrains-contrast"
                    }`}
                >
                  {p + 1}
                </button>
              ),
            )}

            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(page + 1)}
              className="flex h-7 w-7 items-center justify-center border border-transparent hover:bg-jetbrains-gray/10 text-jetbrains-gray-light rounded-sm disabled:opacity-30 transition-colors"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(totalPages - 1)}
              className="flex h-7 w-7 items-center justify-center border border-transparent hover:bg-jetbrains-gray/10 text-jetbrains-gray-light rounded-sm disabled:opacity-30 transition-colors"
            >
              <ChevronsRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
