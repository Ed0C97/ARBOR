#!/bin/bash
# =============================================================================
# A.R.B.O.R. Enterprise - Manual Deployment Script
# Usage: ./scripts/deploy.sh <environment> [image-tag]
# Examples:
#   ./scripts/deploy.sh staging
#   ./scripts/deploy.sh production v1.2.3
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
REGISTRY="us-central1-docker.pkg.dev"
PROJECT_ID="${GCP_PROJECT_ID:-arbor-enterprise}"
REPOSITORY="arbor-enterprise-images"
NAMESPACE="arbor"
GKE_CLUSTER="arbor-enterprise-gke"
GKE_REGION="us-central1"

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
ENVIRONMENT="${1:-}"
IMAGE_TAG="${2:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"

if [ -z "$ENVIRONMENT" ]; then
  echo "Usage: $0 <staging|production> [image-tag]"
  exit 1
fi

if [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
  echo "Error: environment must be 'staging' or 'production'"
  exit 1
fi

REGISTRY_URL="${REGISTRY}/${PROJECT_ID}/${REPOSITORY}"
API_IMAGE="${REGISTRY_URL}/arbor-api:${IMAGE_TAG}"
WORKER_IMAGE="${REGISTRY_URL}/arbor-worker:${IMAGE_TAG}"

echo "======================================================="
echo " A.R.B.O.R. Enterprise - Deployment"
echo "======================================================="
echo " Environment: ${ENVIRONMENT}"
echo " Image tag:   ${IMAGE_TAG}"
echo " API image:   ${API_IMAGE}"
echo " Worker image: ${WORKER_IMAGE}"
echo "======================================================="
echo ""

# -----------------------------------------------------------------------
# Safety checks
# -----------------------------------------------------------------------
if [ "$ENVIRONMENT" = "production" ]; then
  echo "WARNING: You are about to deploy to PRODUCTION"
  read -p "Type 'yes' to continue: " CONFIRM
  if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled"
    exit 0
  fi
fi

# -----------------------------------------------------------------------
# Authenticate and connect to cluster
# -----------------------------------------------------------------------
echo ">> Authenticating to GCP..."
gcloud container clusters get-credentials "${GKE_CLUSTER}" \
  --region "${GKE_REGION}" \
  --project "${PROJECT_ID}"

echo ">> Verifying cluster connection..."
kubectl cluster-info

# -----------------------------------------------------------------------
# Pre-deployment snapshot
# -----------------------------------------------------------------------
echo ""
echo ">> Pre-deployment state:"
kubectl get deployments -n "${NAMESPACE}" -o wide 2>/dev/null || true

# -----------------------------------------------------------------------
# Apply configurations
# -----------------------------------------------------------------------
echo ""
echo ">> Applying configurations..."
kubectl apply -f infrastructure/kubernetes/namespaces/namespace.yaml
kubectl apply -f infrastructure/kubernetes/configmaps/app-config.yaml
kubectl apply -f infrastructure/kubernetes/services/api-service.yaml
kubectl apply -f infrastructure/kubernetes/hpa/api-hpa.yaml

# -----------------------------------------------------------------------
# Deploy API
# -----------------------------------------------------------------------
echo ""
echo ">> Deploying API..."
kubectl set image deployment/arbor-api \
  arbor-api="${API_IMAGE}" \
  run-migrations="${API_IMAGE}" \
  -n "${NAMESPACE}"

kubectl annotate deployment/arbor-api \
  -n "${NAMESPACE}" \
  kubernetes.io/change-cause="Manual deploy ${IMAGE_TAG} by $(whoami) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --overwrite

echo ">> Waiting for API rollout..."
kubectl rollout status deployment/arbor-api \
  -n "${NAMESPACE}" \
  --timeout=300s

# -----------------------------------------------------------------------
# Deploy Worker
# -----------------------------------------------------------------------
echo ""
echo ">> Deploying Worker..."
kubectl set image deployment/arbor-worker \
  arbor-worker="${WORKER_IMAGE}" \
  -n "${NAMESPACE}"

kubectl annotate deployment/arbor-worker \
  -n "${NAMESPACE}" \
  kubernetes.io/change-cause="Manual deploy ${IMAGE_TAG} by $(whoami) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --overwrite

echo ">> Waiting for Worker rollout..."
kubectl rollout status deployment/arbor-worker \
  -n "${NAMESPACE}" \
  --timeout=300s

# -----------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------
echo ""
echo ">> Running health checks..."
API_IP=$(kubectl get svc arbor-api -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

if [ -n "$API_IP" ]; then
  HEALTHY=false
  for i in $(seq 1 10); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${API_IP}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
      echo "   Health check PASSED (attempt ${i})"
      HEALTHY=true
      break
    fi
    echo "   Attempt ${i}/10: HTTP ${HTTP_CODE}, retrying in 10s..."
    sleep 10
  done

  if [ "$HEALTHY" != "true" ]; then
    echo ""
    echo "!! HEALTH CHECK FAILED - Consider rollback:"
    echo "   kubectl rollout undo deployment/arbor-api -n ${NAMESPACE}"
    echo "   kubectl rollout undo deployment/arbor-worker -n ${NAMESPACE}"
    exit 1
  fi
else
  echo "   WARNING: LoadBalancer IP not yet assigned"
fi

# -----------------------------------------------------------------------
# Post-deployment summary
# -----------------------------------------------------------------------
echo ""
echo "======================================================="
echo " Deployment Complete"
echo "======================================================="
echo ""
echo "Pods:"
kubectl get pods -n "${NAMESPACE}" -l part-of=arbor-enterprise -o wide
echo ""
echo "Deployments:"
kubectl get deployments -n "${NAMESPACE}" -o wide
echo ""
echo "Services:"
kubectl get svc -n "${NAMESPACE}"
echo ""
echo "Rollback command:"
echo "  kubectl rollout undo deployment/arbor-api -n ${NAMESPACE}"
echo "  kubectl rollout undo deployment/arbor-worker -n ${NAMESPACE}"
