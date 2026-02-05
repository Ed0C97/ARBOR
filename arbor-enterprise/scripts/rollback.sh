#!/bin/bash
# =============================================================================
# A.R.B.O.R. Enterprise - Rollback Script
# Usage: ./scripts/rollback.sh [revision]
# Examples:
#   ./scripts/rollback.sh        # Rollback to previous version
#   ./scripts/rollback.sh 3      # Rollback to specific revision
# =============================================================================

set -euo pipefail

NAMESPACE="arbor"
REVISION="${1:-}"

echo "======================================================="
echo " A.R.B.O.R. Enterprise - Rollback"
echo "======================================================="

# Show current state
echo ""
echo ">> Current deployment state:"
kubectl get deployments -n "${NAMESPACE}" -o wide
echo ""
echo ">> Deployment history:"
kubectl rollout history deployment/arbor-api -n "${NAMESPACE}"

# Confirm
echo ""
if [ -n "$REVISION" ]; then
  echo "Rolling back to revision: ${REVISION}"
else
  echo "Rolling back to previous version"
fi
read -p "Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Rollback cancelled"
  exit 0
fi

# Rollback API
echo ""
echo ">> Rolling back arbor-api..."
if [ -n "$REVISION" ]; then
  kubectl rollout undo deployment/arbor-api -n "${NAMESPACE}" --to-revision="${REVISION}"
else
  kubectl rollout undo deployment/arbor-api -n "${NAMESPACE}"
fi
kubectl rollout status deployment/arbor-api -n "${NAMESPACE}" --timeout=300s

# Rollback Worker
echo ""
echo ">> Rolling back arbor-worker..."
if [ -n "$REVISION" ]; then
  kubectl rollout undo deployment/arbor-worker -n "${NAMESPACE}" --to-revision="${REVISION}"
else
  kubectl rollout undo deployment/arbor-worker -n "${NAMESPACE}"
fi
kubectl rollout status deployment/arbor-worker -n "${NAMESPACE}" --timeout=300s

# Verify
echo ""
echo ">> Post-rollback state:"
kubectl get pods -n "${NAMESPACE}" -l part-of=arbor-enterprise
echo ""
echo "Rollback complete."
