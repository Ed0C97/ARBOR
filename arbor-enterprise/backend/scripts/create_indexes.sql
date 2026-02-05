-- TIER 2 - Point 5: Advanced Database Indexing Strategy
--
-- PostgreSQL indexes for ARBOR Enterprise
-- Run this script to optimize query performance
--
-- Usage:
--   psql -d arbor_db -f create_indexes.sql

-- ═══════════════════════════════════════════════════════════════════════════
-- ARBOR_ENRICHMENTS table indexes
-- ═══════════════════════════════════════════════════════════════════════════

-- GIN index for JSONB vibe_dna queries (allows filtering by dimensions)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_vibe_dna_gin
ON arbor_enrichments USING GIN (vibe_dna jsonb_path_ops);

-- GIN index for tags array
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_tags_gin
ON arbor_enrichments USING GIN (tags);

-- Composite index for entity lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_entity_lookup
ON arbor_enrichments (entity_type, source_id);

-- Index for sync status queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_neo4j_sync
ON arbor_enrichments (neo4j_synced) WHERE neo4j_synced = false;


-- ═══════════════════════════════════════════════════════════════════════════
-- ARBOR_REVIEW_QUEUE table indexes
-- ═══════════════════════════════════════════════════════════════════════════

-- Status index for queue processing
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_review_queue_status
ON arbor_review_queue (status);

-- Priority index for high-priority items
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_review_queue_priority
ON arbor_review_queue (priority DESC) WHERE status = 'needs_review';

-- Entity lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_review_queue_entity
ON arbor_review_queue (entity_type, source_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- ARBOR_FEEDBACK table indexes
-- ═══════════════════════════════════════════════════════════════════════════

-- User feedback lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_user
ON arbor_feedback (user_id);

-- Entity feedback lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_entity
ON arbor_feedback (entity_type, source_id);

-- Time-based analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_created
ON arbor_feedback (created_at DESC);


-- ═══════════════════════════════════════════════════════════════════════════
-- ARBOR_GOLD_STANDARD table indexes
-- ═══════════════════════════════════════════════════════════════════════════

-- Entity lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gold_standard_entity
ON arbor_gold_standard (entity_type, source_id);

-- Curator queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gold_standard_curator
ON arbor_gold_standard (curated_by);


-- ═══════════════════════════════════════════════════════════════════════════
-- MAGAZINE DB (brands/venues) - READ ONLY
-- Note: These should be run on the magazine_h182 database by the DBA
-- ═══════════════════════════════════════════════════════════════════════════

-- -- Brands full-text search
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brands_search
-- ON brands USING GIN (to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, '')));

-- -- Brands category filter
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brands_category
-- ON brands (category) WHERE is_active = true;

-- -- Brands geospatial (if PostGIS available)
-- -- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brands_geo
-- -- ON brands USING GIST (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326));

-- -- Venues full-text search
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_venues_search
-- ON venues USING GIN (to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, '')));

-- -- Venues category + city filter
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_venues_category_city
-- ON venues (category, city) WHERE is_active = true;

-- -- Venues geospatial
-- -- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_venues_geo
-- -- ON venues USING GIST (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326));


-- ═══════════════════════════════════════════════════════════════════════════
-- ANALYZE tables to update statistics
-- ═══════════════════════════════════════════════════════════════════════════

ANALYZE arbor_enrichments;
ANALYZE arbor_review_queue;
ANALYZE arbor_feedback;
ANALYZE arbor_gold_standard;


-- ═══════════════════════════════════════════════════════════════════════════
-- Verify indexes
-- ═══════════════════════════════════════════════════════════════════════════

SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes
WHERE tablename LIKE 'arbor_%'
ORDER BY tablename, indexname;
