/* -----------------------------------------------------------------------
   API client for the A.R.B.O.R. Enterprise backend.
   All requests go through the NEXT_PUBLIC_API_URL environment variable.

   Data source: magazine_h182 PostgreSQL on Render (read-only).
   Enrichment: arbor_enrichments table (read-write via /admin endpoints).
   ----------------------------------------------------------------------- */

import type {
  DiscoverRequest,
  DiscoverResponse,
  EnrichmentRequest,
  EnrichmentResponse,
  Entity,
  EntityFilters,
  EntityListResponse,
  GraphResponse,
  SearchResponse,
  StatsResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Generic fetch wrapper
// ---------------------------------------------------------------------------

interface FetchOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined | null>;
}

async function apiFetch<T>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const { params, ...init } = options;

  let url = `${API_BASE}/api/v1${path}`;

  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const qs = searchParams.toString();
    if (qs) {
      url += `?${qs}`;
    }
  }

  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...init.headers,
    },
    ...init,
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "Unknown error");
    throw new Error(
      `API error ${response.status}: ${response.statusText} - ${errorBody}`,
    );
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

export async function discover(
  request: DiscoverRequest,
): Promise<DiscoverResponse> {
  return apiFetch<DiscoverResponse>("/discover", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// ---------------------------------------------------------------------------
// Entities — Unified (brands + venues)
// ---------------------------------------------------------------------------

export async function listEntities(
  filters?: EntityFilters,
): Promise<EntityListResponse> {
  return apiFetch<EntityListResponse>("/entities", {
    params: filters as Record<string, string | number | boolean | undefined>,
  });
}

export async function getEntity(id: string): Promise<Entity> {
  return apiFetch<Entity>(`/entities/${id}`);
}

// ---------------------------------------------------------------------------
// Brands only
// ---------------------------------------------------------------------------

export async function listBrands(
  filters?: Omit<EntityFilters, "entity_type" | "city">,
): Promise<EntityListResponse> {
  return apiFetch<EntityListResponse>("/brands", {
    params: filters as Record<string, string | number | boolean | undefined>,
  });
}

// ---------------------------------------------------------------------------
// Venues only
// ---------------------------------------------------------------------------

export async function listVenues(
  filters?: Omit<EntityFilters, "entity_type">,
): Promise<EntityListResponse> {
  return apiFetch<EntityListResponse>("/venues", {
    params: filters as Record<string, string | number | boolean | undefined>,
  });
}

// ---------------------------------------------------------------------------
// Admin — Stats
// ---------------------------------------------------------------------------

export async function getStats(): Promise<StatsResponse> {
  return apiFetch<StatsResponse>("/admin/stats");
}

// ---------------------------------------------------------------------------
// Admin — Enrichment
// ---------------------------------------------------------------------------

export async function enrichEntity(
  entityId: string,
  body: EnrichmentRequest,
): Promise<EnrichmentResponse> {
  return apiFetch<EnrichmentResponse>(`/admin/enrich/${entityId}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function deleteEnrichment(
  entityId: string,
): Promise<{ detail: string }> {
  return apiFetch<{ detail: string }>(`/admin/enrich/${entityId}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Vector Search
// ---------------------------------------------------------------------------

export async function vectorSearch(
  query: string,
  options?: { category?: string; city?: string; limit?: number },
): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/search/vector", {
    params: { query, ...options },
  });
}

// ---------------------------------------------------------------------------
// Knowledge Graph
// ---------------------------------------------------------------------------

export async function getRelatedEntities(
  entityName: string,
): Promise<GraphResponse> {
  return apiFetch<GraphResponse>("/graph/related", {
    params: { entity_name: entityName },
  });
}

export async function getLineage(
  entityName: string,
  depth?: number,
): Promise<GraphResponse> {
  return apiFetch<GraphResponse>("/graph/lineage", {
    params: { entity_name: entityName, depth },
  });
}

export async function getBrandRetailers(
  brandName: string,
  city?: string,
): Promise<GraphResponse> {
  return apiFetch<GraphResponse>("/graph/brand-retailers", {
    params: { brand_name: brandName, city },
  });
}
