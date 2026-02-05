#!/bin/bash
# =============================================================================
# A.R.B.O.R. Enterprise - Database Backup Script
# Usage: ./scripts/backup_db.sh [output-dir]
# =============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAMESPACE="arbor"

echo "=== A.R.B.O.R. Database Backup ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------
# PostgreSQL Backup
# -----------------------------------------------------------------------
echo ">> Backing up PostgreSQL..."
POSTGRES_POD=$(kubectl get pods -n "${NAMESPACE}" -l app=postgres -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -n "$POSTGRES_POD" ]; then
  kubectl exec -n "${NAMESPACE}" "${POSTGRES_POD}" -- \
    pg_dump -U arbor_admin -d arbor --format=custom \
    > "${OUTPUT_DIR}/postgres_${TIMESTAMP}.dump"
  echo "   PostgreSQL backup saved to ${OUTPUT_DIR}/postgres_${TIMESTAMP}.dump"
else
  echo "   WARNING: PostgreSQL pod not found, trying local connection..."
  PGPASSWORD="${POSTGRES_PASSWORD:-}" pg_dump \
    -h "${POSTGRES_HOST:-localhost}" \
    -p "${POSTGRES_PORT:-5432}" \
    -U "${POSTGRES_USER:-arbor_admin}" \
    -d "${POSTGRES_DB:-arbor}" \
    --format=custom \
    > "${OUTPUT_DIR}/postgres_${TIMESTAMP}.dump" 2>/dev/null || echo "   PostgreSQL backup skipped (not available)"
fi

# -----------------------------------------------------------------------
# Neo4j Backup
# -----------------------------------------------------------------------
echo ">> Backing up Neo4j..."
NEO4J_POD=$(kubectl get pods -n "${NAMESPACE}" -l app=neo4j -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -n "$NEO4J_POD" ]; then
  kubectl exec -n "${NAMESPACE}" "${NEO4J_POD}" -- \
    neo4j-admin database dump neo4j --to-path=/tmp/
  kubectl cp "${NAMESPACE}/${NEO4J_POD}:/tmp/neo4j.dump" \
    "${OUTPUT_DIR}/neo4j_${TIMESTAMP}.dump"
  echo "   Neo4j backup saved to ${OUTPUT_DIR}/neo4j_${TIMESTAMP}.dump"
else
  echo "   WARNING: Neo4j pod not found, skipping"
fi

# -----------------------------------------------------------------------
# Qdrant Snapshot
# -----------------------------------------------------------------------
echo ">> Creating Qdrant snapshot..."
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

SNAPSHOT_RESPONSE=$(curl -s -X POST "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/arbor_entities/snapshots" 2>/dev/null || echo "")
if [ -n "$SNAPSHOT_RESPONSE" ]; then
  echo "   Qdrant snapshot created: ${SNAPSHOT_RESPONSE}"
else
  echo "   WARNING: Qdrant not reachable, skipping"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "=== Backup Complete ==="
echo "Files:"
ls -la "${OUTPUT_DIR}"/*_${TIMESTAMP}* 2>/dev/null || echo "   No backup files created"
echo ""
echo "Restore commands:"
echo "  PostgreSQL: pg_restore -d arbor ${OUTPUT_DIR}/postgres_${TIMESTAMP}.dump"
echo "  Neo4j:      neo4j-admin database load neo4j --from-path=/path/to/neo4j_${TIMESTAMP}.dump"
