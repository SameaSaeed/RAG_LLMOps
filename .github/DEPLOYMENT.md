# GitHub Actions & Minikube Deployment Guide

## Overview

This project includes automated CI/CD pipelines using GitHub Actions to test, build, and deploy to Minikube. The workflows provide:

- **Automated Testing**: Unit and integration tests on every push
- **Code Quality**: Linting, formatting, security scanning
- **Docker Builds**: Automated image building and pushing to GitHub Container Registry
- **Kubernetes Validation**: K8s manifest validation and security scanning
- **Minikube Deployment**: Automated deployment to Minikube for integration testing

## Workflows

### 1. `deploy.yml` - Test and Deploy LLMOps

**Triggers**: 
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**Jobs**:

#### Test Job
```bash
- Python 3.13 setup
- Dependency installation
- Code quality checks (Flake8, Black, isort)
- Run pytest test suite
- Upload coverage reports to Codecov
```

#### Build Job (on main/develop push only)
```bash
- Docker Buildx setup
- Login to GitHub Container Registry
- Build and push llmops-api image
```

#### Deploy to Minikube Job (on main push only)
```bash
- Start Minikube cluster (v1.36.0, K8s v1.32.0)
- Build Docker image locally
- Create namespace and secrets
- Apply K8s ConfigMaps, ServiceAccounts, Deployments
- Apply K8s Services
- Wait for rollout (10m timeout)
- Verify service accessibility
- Fetch logs on failure
```

### 2. `lint.yml` - Code Quality & Linting

**Triggers**: Push/PR to main or develop

**Checks**:
- Flake8 linting
- Black formatting check
- isort import sorting
- Bandit security audit
- mypy type checking
- pip-audit vulnerability scan
- Safety vulnerability check

### 3. `docker.yml` - Build and Push Docker Images

**Triggers**: 
- Push to main/develop
- Push tags (v*)
- Manual workflow dispatch

**Features**:
- Builds both API and UI images
- Multi-platform support via Docker Buildx
- Image scanning with Trivy
- Automatic tagged releases

### 4. `k8s-validate.yml` - Kubernetes Manifests Validation

**Triggers**: Changes to K8s files or workflow

**Validations**:
- kubeval manifest validation
- YAML linting with yamllint
- Kubernetes dry-run
- Kube-score analysis
- Kubesec security scanning
- Polaris compliance check

## Setting Up GitHub Secrets

To enable full CI/CD functionality, add these secrets to your GitHub repository:

**Go to**: Settings → Secrets and variables → Actions

### Required Secrets

```
GROQ_API_KEY            # Groq API key for LLM
ASTRA_DB_TOKEN          # AstraDB application token
ASTRA_DB_ID             # AstraDB database ID
CODECOV_TOKEN           # (Optional) Codecov token for coverage uploads
```

**To add a secret via GitHub CLI**:
```bash
gh secret set GROQ_API_KEY --body "your-key-here"
gh secret set ASTRA_DB_TOKEN --body "your-token-here"
gh secret set ASTRA_DB_ID --body "your-db-id-here"
```

## Local Development with Minikube

### Prerequisites

```bash
# Install Minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Docker (required for Minikube driver)
sudo apt-get install docker.io
```

### Starting Minikube

```bash
# Start cluster with 4GB memory and 4 CPUs
minikube start --memory=4096 --cpus=4 --driver=docker

# Verify cluster
minikube status
kubectl get nodes
```

### Building Images

```bash
# Evaluate Minikube Docker environment
eval $(minikube -p minikube docker-env)

# Build images
docker build -t llmops-api:latest .

# Verify
docker images | grep llmops
```

### Deploying Locally

```bash
# Apply K8s manifests
cd K8s

# Create namespace
kubectl create namespace llmops

# Create secrets
kubectl create secret generic llmops-secrets \
  --from-literal=groq-api-key='test-key' \
  --from-literal=astra-token='test-token' \
  --from-literal=astra-db-id='test-id' \
  -n llmops

# Apply all configurations
kubectl apply -f config.yaml
kubectl apply -f serviceaccount.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods -n llmops
kubectl get svc -n llmops
```

### Accessing Services

```bash
# Port forward to API
kubectl port-forward -n llmops svc/llmops-api 8000:8000

# In another terminal, port forward to UI
kubectl port-forward -n llmops svc/llmops-ui 8501:8501

# Access services
# API: http://localhost:8000/docs
# UI: http://localhost:8501
```

### Viewing Logs

```bash
# API logs
kubectl logs -n llmops -l app=llmops-api -f

# UI logs
kubectl logs -n llmops -l app=llmops-ui -f

# Get logs from specific pod
kubectl logs -n llmops <pod-name>
```

## Running Tests Locally

### Unit Tests
```bash
python -m pytest tests/ -v
```

### With Coverage
```bash
python -m pytest tests/ --cov=src --cov=utils --cov=configs --cov-report=html
```

### Specific Test
```bash
python -m pytest tests/test_RAG.py::test_invoke_raises_when_chain_not_initialized -v
```

## Docker Compose Alternative

For local testing without Minikube:

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api
docker-compose logs -f ui

# Stop services
docker-compose down
```

## Troubleshooting

### Minikube won't start
```bash
# Reset Minikube
minikube delete
minikube start --memory=4096 --cpus=4

# Check Docker daemon
docker ps

# Verify permissions
sudo usermod -aG docker $USER
```

### Pods not running
```bash
# Describe pod for events
kubectl describe pod <pod-name> -n llmops

# Check resource availability
kubectl top nodes
kubectl top pods -n llmops

# Check resource limits
kubectl describe deployment llmops-api -n llmops
```

### Image pull errors
```bash
# Verify image exists
docker images | grep llmops

# Rebuild if needed
eval $(minikube -p minikube docker-env)
docker build -t llmops-api:latest .
```

### High memory/CPU usage
```bash
# Increase Minikube resources
minikube delete
minikube start --memory=6144 --cpus=6

# Check deployment resources
kubectl get pods -n llmops -o json | jq '.items[] | {name: .metadata.name, cpu: .spec.containers[].resources}'
```

## CI/CD Pipeline Flow

```
┌─────────────────────────────────────┐
│   Push to main/develop              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Run Tests (test.yml)              │
│   - Python 3.13 setup               │
│   - Dependencies                    │
│   - Code quality checks             │
│   - Pytest suite                    │
└──────────────┬──────────────────────┘
               │
          ┌────┴─────┐
          │           │
          ▼           ▼
    ┌──────────┐  ┌──────────────┐
    │ Success  │  │ Failure      │
    │ Continue │  │ Stop         │
    └────┬─────┘  └──────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Build & Push (docker.yml)         │
│   - Build API image                 │
│   - Build UI image                  │
│   - Push to GHCR                    │
│   - Security scan (Trivy)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Deploy to Minikube (deploy.yml)   │
│   - Create namespace                │
│   - Create secrets                  │
│   - Apply K8s manifests             │
│   - Wait for rollout                │
│   - Verify services                 │
└──────────────┬──────────────────────┘
               │
          ┌────┴─────┐
          │           │
          ▼           ▼
    ┌──────────┐  ┌──────────────┐
    │ Success  │  │ Failure      │
    │ Notify   │  │ Fetch Logs   │
    │ (Email)  │  │ Notify       │
    └──────────┘  └──────────────┘
```

## Best Practices

1. **Always test locally** before pushing:
   ```bash
   python -m pytest tests/
   docker-compose up
   ```

2. **Use feature branches** for development:
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Add meaningful commit messages**:
   ```bash
   git commit -m "feat: add new feature" # Follows conventional commits
   ```

4. **Keep dependencies updated**:
   ```bash
   pip install -U -r requirements.txt
   ```

5. **Monitor workflow runs** in GitHub Actions tab

6. **Review security warnings** from Trivy and Bandit

## Advanced Configuration

### Custom Test Matrix
Edit `deploy.yml` to test multiple Python versions:
```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12', '3.13']
```

### Custom Minikube Resources
Edit `deploy.yml` minikube step:
```yaml
start-args: '--memory=8192 --cpus=8 --disk-size=50g'
```

### Conditional Deployments
Deploy only on version tags:
```yaml
if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
```

## Documentation Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Minikube Documentation](https://minikube.sigs.k8s.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
