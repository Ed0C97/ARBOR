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
        <h1 className="text-2xl font-semibold text-gray-900">Entity Management</h1>
        <p className="mt-1 text-sm text-gray-500">
          {total} entities total &mdash; browse, search, and manage the knowledge graph
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 border border-gray-200 bg-white p-4 shadow-render">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Search by name..."
            className="flex h-9 w-full border border-gray-300 bg-white pl-10 pr-3 text-sm text-gray-900 placeholder:text-gray-400 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20 transition-all"
          />
        </div>

        <Filter className="h-4 w-4 text-gray-400" />

        <select
          value={entityType}
          onChange={(e) => {
            setEntityType(e.target.value as EntityType | "all");
            setPage(0);
          }}
          className="h-9 border border-gray-300 bg-white px-3 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20"
        >
          {ENTITY_TYPES.map((t) => (
            <option key={t} value={t}>
              {t === "all" ? "All Types" : t.charAt(0).toUpperCase() + t.slice(1) + "s"}
            </option>
          ))}
        </select>

        <select
          value={category}
          onChange={(e) => {
            setCategory(e.target.value);
            setPage(0);
          }}
          className="h-9 border border-gray-300 bg-white px-3 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20"
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c === "all"
                ? "All Categories"
                : c.charAt(0).toUpperCase() + c.slice(1).replace("_", " ")}
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div className="rounded border border-gray-200 bg-white overflow-hidden shadow-render">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-gray-200 bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Name</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Type</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Category</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">City</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Price</th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-gray-500">Rating</th>
              <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Actions</th>
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
                  No entities found
                </td>
              </tr>
            ) : (
              entities.map((entity) => (
                <tr key={entity.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900 text-sm">
                        {entity.name}
                      </span>
                      {entity.is_featured && (
                        <Star className="h-3 w-3 shrink-0 text-amber-500 fill-amber-100" />
                      )}
                      {entity.verified && (
                        <span className="bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 border border-blue-200 rounded-md">
                          Verified
                        </span>
                      )}
                    </div>
                    {entity.instagram && (
                      <div className="mt-0.5 flex items-center gap-1 text-xs text-gray-400">
                        <Instagram className="h-3 w-3" />
                        {entity.instagram}
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
                  <td className="px-4 py-3 text-gray-900 text-sm">
                    {entity.rating ? entity.rating.toFixed(1) : "\u2014"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end">
                      <Link
                        href={`/entity/${entity.id}`}
                        className="p-2 text-gray-400 hover:bg-gray-100 hover:text-[#4353FF] transition-colors"
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
        <div className="flex flex-col items-center justify-between gap-4 sm:flex-row pt-2">
          <span className="text-sm text-gray-500">
            Showing <span className="font-medium text-gray-900">{rangeStart}&ndash;{rangeEnd}</span> of {total}
          </span>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Rows:</span>
            <select
              value={limit}
              onChange={(e) => {
                setLimit(Number(e.target.value));
                setPage(0);
              }}
              className="h-8 border border-gray-300 bg-white px-2 text-sm text-gray-700 focus:border-[#4353FF] focus:outline-none"
            >
              {ROWS_PER_PAGE_OPTIONS.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

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
