# LLMOps Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (1.24+)
- `kubectl` CLI configured
- Docker images built and pushed to registry:
  - `llmops-api:latest`
  - `llmops-ui:latest`

## Setup

### 1. Create Namespace and Secrets

```bash
# Apply all resources in order
kubectl apply -f K8s/config.yaml
kubectl apply -f K8s/serviceaccount.yaml
kubectl apply -f K8s/secret.yaml  # UPDATE SECRETS FIRST!
```

### 2. Update Secret Values

Before deploying, set your actual credentials:

```bash
# Edit secret with your real values
kubectl edit secret llmops-secrets -n llmops

# Or create it from command line:
kubectl create secret generic llmops-secrets \
  --from-literal=groq-api-key='your-key' \
  --from-literal=astra-token='your-token' \
  --from-literal=astra-db-id='your-db-id' \
  -n llmops
```

### 3. Deploy Applications

```bash
# Deploy API and UI
kubectl apply -f K8s/deployment.yaml

# Create services
kubectl apply -f K8s/service.yaml

# Enable auto-scaling
kubectl apply -f K8s/hpa.yaml

# (Optional) Setup Ingress for external access
kubectl apply -f K8s/ingress.yaml
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n llmops

# Check services
kubectl get svc -n llmops

# View logs
kubectl logs -n llmops -l app=llmops-api -f
kubectl logs -n llmops -l app=llmops-ui -f

# Check HPA status
kubectl get hpa -n llmops
```

## Accessing Services

### Local/Minikube
```bash
# API
kubectl port-forward -n llmops svc/llmops-api 8000:8000

# UI
kubectl port-forward -n llmops svc/llmops-ui 8501:8501
```

### NodePort (default)
- API: http://node-ip:30800
- UI: http://node-ip:30801

### Ingress (if configured)
- Update `/etc/hosts` or DNS to point `llmops.example.com` to ingress IP
- Access: https://llmops.example.com

## Building Docker Images

```bash
# Build and tag images
docker build -t llmops-api:latest .
docker build -t llmops-ui:latest .

# For Minikube (no push needed)
minikube image load llmops-api:latest
minikube image load llmops-ui:latest

# For external registry
docker tag llmops-api:latest your-registry/llmops-api:latest
docker push your-registry/llmops-api:latest
```

## Monitoring & Troubleshooting

```bash
# View resource usage
kubectl top pods -n llmops

# Describe pod (troubleshoot issues)
kubectl describe pod <pod-name> -n llmops

# Get deployment status
kubectl rollout status deployment/llmops-api -n llmops

# Restart deployment
kubectl rollout restart deployment/llmops-api -n llmops

# View events
kubectl get events -n llmops --sort-by='.lastTimestamp'
```

## Configuration Management

### Update ConfigMap
```bash
kubectl apply -f K8s/config.yaml
# Restart pods to pick up changes
kubectl rollout restart deployment/llmops-api -n llmops
```

### Update Secrets
```bash
kubectl delete secret llmops-secrets -n llmops
kubectl create secret generic llmops-secrets \
  --from-literal=groq-api-key='new-key' \
  --from-literal=astra-token='new-token' \
  --from-literal=astra-db-id='new-db-id' \
  -n llmops
# Restart pods
kubectl rollout restart deployment/llmops-api -n llmops
```

## Cleanup

```bash
# Delete all resources
kubectl delete namespace llmops

# Or selective deletion
kubectl delete -f K8s/deployment.yaml -n llmops
kubectl delete -f K8s/service.yaml -n llmops
kubectl delete -f K8s/hpa.yaml -n llmops
```

## Auto-Scaling

The HPAs are configured to:
- **API**: Scale from 2 to 5 replicas based on CPU (70%) and memory (80%) utilization
- **UI**: Scale from 1 to 3 replicas based on CPU (75%) and memory (85%) utilization

Monitor scaling events:
```bash
kubectl describe hpa llmops-api-hpa -n llmops
```

## Multi-Cluster Deployment

For production multi-cluster setup, use:
- **GitOps**: ArgoCD or Flux
- **Service Mesh**: Istio or Linkerd for traffic management
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack or Loki

## Production Checklist

- [ ] Update secret values with real credentials
- [ ] Configure Ingress domain and SSL certificates
- [ ] Set up log aggregation
- [ ] Configure monitoring and alerts
- [ ] Set resource quotas per pod
- [ ] Enable Pod Security Policies
- [ ] Review RBAC permissions
- [ ] Test failover scenarios
- [ ] Configure backup for persistent data
- [ ] Document disaster recovery process
