/* -----------------------------------------------------------------------
   TypeScript types for the A.R.B.O.R. Enterprise frontend.
   These mirror the Pydantic response models defined in the FastAPI backend.

   Data source: magazine_h182 PostgreSQL (brands + venues tables)
   ----------------------------------------------------------------------- */

// ---------------------------------------------------------------------------
// Entity types
// ---------------------------------------------------------------------------

export type EntityType = "brand" | "venue";

// ---------------------------------------------------------------------------
// Vibe DNA — ARBOR enrichment layer
// ---------------------------------------------------------------------------

export interface VibeDimensions {
  formality: number;
  craftsmanship: number;
  price_value: number;
  atmosphere: number;
  service_quality: number;
  exclusivity: number;
  [key: string]: number;
}

export interface VibeDNA {
  dimensions: VibeDimensions;
  tags: string[];
  signature_items: string[];
  target_audience: string;
  summary: string;
}

// ---------------------------------------------------------------------------
// Entity — Unified brand + venue representation
// ---------------------------------------------------------------------------

export interface Entity {
  /** Composite ID: "brand_42" or "venue_17" */
  id: string;
  entity_type: EntityType;
  source_id: number;
  name: string;
  slug: string;
  category: string;
  city: string | null;
  region: string | null;
  country: string | null;
  address: string | null;
  latitude: number | null;
  longitude: number | null;
  maps_url: string | null;
  website: string | null;
  instagram: string | null;
  email: string | null;
  phone: string | null;
  contact_person: string | null;
  description: string | null;
  specialty: string | null;
  notes: string | null;
  gender: string | null;
  style: string | null;
  rating: number | null;
  price_range: string | null;
  is_featured: boolean;
  is_active: boolean;
  priority: number | null;
  verified: boolean | null;
  // ARBOR enrichment
  vibe_dna: Record<string, unknown> | null;
  tags: string[] | null;
  // Audit
  created_at: string | null;
  updated_at: string | null;
}

export interface EntityListResponse {
  items: Entity[];
  total: number;
  offset: number;
  limit: number;
}

// ---------------------------------------------------------------------------
// Entity filters (mirrors backend query params)
// ---------------------------------------------------------------------------

export interface EntityFilters {
  entity_type?: EntityType;
  category?: string;
  city?: string;
  country?: string;
  gender?: string;
  style?: string;
  is_active?: boolean;
  is_featured?: boolean;
  search?: string;
  offset?: number;
  limit?: number;
}

// ---------------------------------------------------------------------------
// Admin — Stats
// ---------------------------------------------------------------------------

export interface StatsResponse {
  total_entities: number;
  total_brands: number;
  total_venues: number;
  enriched_entities: number;
}

// ---------------------------------------------------------------------------
// Admin — Enrichment
// ---------------------------------------------------------------------------

export interface EnrichmentRequest {
  vibe_dna?: Record<string, unknown>;
  tags?: string[];
}

export interface EnrichmentResponse {
  entity_id: string;
  entity_type: EntityType;
  source_id: number;
  vibe_dna: Record<string, unknown> | null;
  tags: string[] | null;
  message: string;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

export interface DiscoverRequest {
  query: string;
  location?: string | null;
  category?: string | null;
  price_max?: number | null;
  limit?: number;
}

export interface RecommendationItem {
  id: string;
  name: string;
  score: number;
  category: string;
  city: string;
  tags: string[];
  dimensions: VibeDimensions | Record<string, never>;
}

export interface DiscoverResponse {
  recommendations: RecommendationItem[];
  response_text: string;
  confidence: number;
  query_intent: string;
  latency_ms: number;
  cached: boolean;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

export interface SearchResultItem {
  id: string;
  name: string;
  score: number;
  category: string;
  city: string;
  tags: string[];
  dimensions: VibeDimensions | Record<string, never>;
}

export interface SearchResponse {
  results: SearchResultItem[];
  total: number;
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export interface GraphRelation {
  type: string;
  name: string;
  category: string;
  city: string;
  extra: Record<string, unknown>;
}

export interface GraphResponse {
  results: GraphRelation[];
  total: number;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  recommendations?: RecommendationItem[];
  timestamp: Date;
}
