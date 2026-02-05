"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import {
  Search,
  Eye,
  Loader2,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Star,
  Instagram,
  Filter
} from "lucide-react";
import Link from "next/link";
import type { Entity, EntityListResponse, EntityFilters, EntityType } from "@/lib/types";
import { listEntities } from "@/lib/api";

const ENTITY_TYPES: Array<EntityType | "all"> = ["all", "brand", "venue"];

const CATEGORIES = [
  "all",
  "tailoring",
  "accessories",
  "clothing",
  "footwear",
  "jewelry",
  "food_drink",
  "restaurant",
  "bar",
  "hotel",
  "shop",
  "cafe",
  "gallery",
  "spa",
];

const ROWS_PER_PAGE_OPTIONS = [10, 15, 20, 25, 50];

export default function EntitiesAdminPage() {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [entityType, setEntityType] = useState<EntityType | "all">("all");
  const [category, setCategory] = useState("all");
  const [page, setPage] = useState(0);
  const [limit, setLimit] = useState(20);

  const fetchEntities = useCallback(async () => {
    setLoading(true);
    try {
      const filters: EntityFilters = {
        offset: page * limit,
        limit,
      };
      if (entityType !== "all") filters.entity_type = entityType;
      if (category !== "all") filters.category = category;
      if (search.trim()) filters.search = search.trim();

      const data: EntityListResponse = await listEntities(filters);
      setEntities(data.items);
      setTotal(data.total);
    } catch {
      setEntities([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, limit, entityType, category, search]);

  useEffect(() => {
    fetchEntities();
  }, [fetchEntities]);

  const totalPages = Math.ceil(total / limit);

  const rangeStart = total === 0 ? 0 : page * limit + 1;
  const rangeEnd = Math.min((page + 1) * limit, total);

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
      // always show first page
      pages.push(0);
      if (page > 3) pages.push("ellipsis-start");

      const start = Math.max(1, page - 2);
      const end = Math.min(totalPages - 2, page + 2);
      for (let i = start; i <= end; i++) pages.push(i);

      if (page < totalPages - 4) pages.push("ellipsis-end");
      // always show last page
      pages.push(totalPages - 1);
    }
    return pages;
  }, [page, totalPages]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-jetbrains-contrast">Entity Management</h1>
        <p className="text-sm text-jetbrains-gray-light font-mono">
          &gt; {total} entities total &mdash; data source: magazine_h182
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 bg-jetbrains-dark border border-jetbrains-gray/30 p-4 rounded-sm">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-jetbrains-gray-light" />
          <input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Search by name..."
            className="flex h-8 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink pl-9 pr-3 text-xs text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
          />
        </div>

        <Filter className="h-4 w-4 text-jetbrains-gray-light" />

        {/* Entity type filter */}
        <select
          value={entityType}
          onChange={(e) => {
            setEntityType(e.target.value as EntityType | "all");
            setPage(0);
          }}
          className="h-8 rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-xs font-mono text-jetbrains-contrast focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue"
        >
          {ENTITY_TYPES.map((t) => (
            <option key={t} value={t}>
              [{t === "all" ? "All Types" : t.charAt(0).toUpperCase() + t.slice(1) + "s"}]
            </option>
          ))}
        </select>

        {/* Category filter */}
        <select
          value={category}
          onChange={(e) => {
            setCategory(e.target.value);
            setPage(0);
          }}
          className="h-8 rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-xs font-mono text-jetbrains-contrast focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue"
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              [{c === "all"
                ? "All Categories"
                : c.charAt(0).toUpperCase() + c.slice(1).replace("_", " ")}]
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div className="rounded-sm border border-jetbrains-gray/30 bg-jetbrains-dark overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-jetbrains-gray/30 bg-jetbrains-ink">
            <tr>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                Name
              </th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                Type
              </th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                Category
              </th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                City
              </th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                Price
              </th>
              <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                Rating
              </th>
              <th className="px-4 py-3 text-right font-mono text-[10px] uppercase tracking-wider text-jetbrains-gray-light font-medium">
                Actions
              </th>
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
                  &lt; No entities found &gt;
                </td>
              </tr>
            ) : (
              entities.map((entity) => (
                <tr
                  key={entity.id}
                  className="hover:bg-jetbrains-gray/5 transition-colors group"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-jetbrains-contrast font-sans text-sm">
                        {entity.name}
                      </span>
                      {entity.is_featured && (
                        <Star className="h-3 w-3 shrink-0 text-jetbrains-orange fill-jetbrains-orange/20" />
                      )}
                      {entity.verified && (
                        <span className="bg-jetbrains-blue/10 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider text-jetbrains-blue border border-jetbrains-blue/20 rounded-sm">
                          Verified
                        </span>
                      )}
                    </div>
                    {entity.instagram && (
                      <div className="mt-0.5 flex items-center gap-1 text-xs text-jetbrains-gray-light font-mono">
                        <Instagram className="h-3 w-3" />
                        {entity.instagram}
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
                  <td className="px-4 py-3 capitalize text-jetbrains-gray-light text-sm">
                    {entity.category}
                  </td>
                  <td className="px-4 py-3 text-jetbrains-gray-light text-sm">
                    {entity.city || "\u2014"}
                  </td>
                  <td className="px-4 py-3 font-mono text-jetbrains-gray-light text-xs">
                    {entity.price_range !== null ? "$".repeat(Number(entity.price_range)) : "\u2014"}
                  </td>
                  <td className="px-4 py-3 font-mono text-jetbrains-contrast text-xs">
                    {entity.rating ? entity.rating.toFixed(1) : "\u2014"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <Link
                        href={`/entity/${entity.id}`}
                        className="p-1.5 text-jetbrains-gray-light hover:bg-jetbrains-ink hover:text-jetbrains-blue rounded-sm"
                      >
                        <Eye className="h-4 w-4" />
                      </Link>
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

          {/* Center: rows per page */}
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-jetbrains-gray-light">
              Rows/page:
            </span>
            <select
              value={limit}
              onChange={(e) => {
                setLimit(Number(e.target.value));
                setPage(0);
              }}
              className="h-6 border border-jetbrains-gray/50 bg-jetbrains-ink px-1 font-mono text-xs text-jetbrains-contrast focus:border-jetbrains-blue focus:outline-none rounded-sm"
            >
              {ROWS_PER_PAGE_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

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
