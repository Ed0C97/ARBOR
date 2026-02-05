// TIER 2 - Point 5: Neo4j Database Indexing Strategy
//
// Run these commands in Neo4j Browser or via cypher-shell
//
// Usage:
//   cypher-shell -u neo4j -p <password> -f neo4j_indexes.cypher

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRAINTS (also create unique indexes)
// ═══════════════════════════════════════════════════════════════════════════

// Unique constraint on Entity.id
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Unique constraint on Entity.slug
CREATE CONSTRAINT entity_slug_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.slug IS UNIQUE;

// Unique constraint on Style.name
CREATE CONSTRAINT style_name_unique IF NOT EXISTS
FOR (s:Style) REQUIRE s.name IS UNIQUE;

// Unique constraint on AbstractEntity.name
CREATE CONSTRAINT abstract_entity_name_unique IF NOT EXISTS
FOR (a:AbstractEntity) REQUIRE a.name IS UNIQUE;


// ═══════════════════════════════════════════════════════════════════════════
// SINGLE-PROPERTY INDEXES
// ═══════════════════════════════════════════════════════════════════════════

// Entity category index for filtering
CREATE INDEX entity_category_index IF NOT EXISTS
FOR (e:Entity) ON (e.category);

// Entity city index for location-based queries
CREATE INDEX entity_city_index IF NOT EXISTS
FOR (e:Entity) ON (e.city);

// Entity country index
CREATE INDEX entity_country_index IF NOT EXISTS
FOR (e:Entity) ON (e.country);

// Entity type index (brand/venue)
CREATE INDEX entity_type_index IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

// Entity active status
CREATE INDEX entity_active_index IF NOT EXISTS
FOR (e:Entity) ON (e.is_active);


// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITE INDEXES
// ═══════════════════════════════════════════════════════════════════════════

// Category + City composite for common queries
CREATE INDEX entity_category_city_index IF NOT EXISTS
FOR (e:Entity) ON (e.category, e.city);

// Type + Category composite
CREATE INDEX entity_type_category_index IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type, e.category);


// ═══════════════════════════════════════════════════════════════════════════
// FULL-TEXT SEARCH INDEXES
// ═══════════════════════════════════════════════════════════════════════════

// Full-text index on Entity name and description
CREATE FULLTEXT INDEX entity_text_search IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.description];

// Full-text index on Style properties
CREATE FULLTEXT INDEX style_text_search IF NOT EXISTS
FOR (s:Style) ON EACH [s.name, s.description];


// ═══════════════════════════════════════════════════════════════════════════
// RELATIONSHIP INDEXES (Neo4j 5.x+)
// ═══════════════════════════════════════════════════════════════════════════

// Index on TRAINED_BY relationship for lineage queries
// CREATE INDEX trained_by_index IF NOT EXISTS
// FOR ()-[r:TRAINED_BY]-() ON (r.year);

// Index on HAS_STYLE relationship
// CREATE INDEX has_style_index IF NOT EXISTS
// FOR ()-[r:HAS_STYLE]-() ON (r.strength);


// ═══════════════════════════════════════════════════════════════════════════
// VERIFY INDEXES
// ═══════════════════════════════════════════════════════════════════════════

SHOW INDEXES;
