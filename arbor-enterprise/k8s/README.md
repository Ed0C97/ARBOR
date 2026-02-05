# ARBOR Enterprise - Kubernetes Manifests
# TIER 11 - Infrastructure Hardening

# These manifests are organized as follows:
# - base/ - Core application deployment
# - networking/ - Service mesh, network policies
# - security/ - Pod security, RBAC
# - monitoring/ - Prometheus, alerts

# This README documents the infrastructure components.

## Directory Structure

```
k8s/
├── base/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secrets.yaml (external-secrets)
├── networking/
│   ├── linkerd/           # TIER 11 - Point 57
│   ├── network-policies/  # TIER 11 - Point 62
│   └── ingress.yaml
├── databases/
│   ├── pgbouncer/         # TIER 11 - Point 58
│   └── redis.yaml
├── security/
│   ├── pod-security.yaml  # TIER 11 - Point 63
│   └── rbac.yaml
└── monitoring/
    ├── prometheus/
    └── grafana/
```

## Quick Setup

```bash
# Apply base resources
kubectl apply -k k8s/base/

# Enable Linkerd service mesh (TIER 11 - Point 57)
linkerd inject k8s/base/deployment.yaml | kubectl apply -f -

# Apply network policies (TIER 11 - Point 62)
kubectl apply -f k8s/networking/network-policies/

# Apply pod security (TIER 11 - Point 63)
kubectl apply -f k8s/security/pod-security.yaml
```

## Multi-Zone Availability (TIER 11 - Point 60)

See deployment.yaml for:
- Pod anti-affinity rules
- Topology spread constraints
- Zone-aware scheduling
