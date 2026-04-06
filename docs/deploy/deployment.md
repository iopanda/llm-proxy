# Deployment Guide

## Docker

### Build

```bash
docker build -t llm-proxy:latest .
```

### Run

```bash
docker run -p 8080:8080 \
  -v $(pwd)/config/config.yaml:/app/config/config.yaml \
  llm-proxy:latest
```

## Kubernetes with Helm

### Install

```bash
helm install llm-proxy ./deployments/helm/llm-proxy \
  --set config.models[0].mapped=gpt-4-proxy \
  --set config.models[0].source=gpt-4 \
  --set config.models[0].protocol=openai \
  --set config.models[0].endpoint=https://api.openai.com \
  --set config.models[0].access_token=sk-your-token
```

Alternatively, use a `values.yaml` override:

```yaml
# my-values.yaml
config:
  server:
    port: 8080
  models:
    - mapped: gpt-4-proxy
      source: gpt-4
      protocol: openai
      endpoint: https://api.openai.com
      access_token: ""        # use Kubernetes Secret instead
```

```bash
helm install llm-proxy ./deployments/helm/llm-proxy -f my-values.yaml
```

### Managing Secrets

For production, avoid putting `access_token` in `values.yaml`. Instead:

1. Create a Kubernetes Secret:
   ```bash
   kubectl create secret generic llm-proxy-tokens \
     --from-literal=openai-token=sk-your-token
   ```

2. Mount the secret as an environment variable and reference it from a custom config, or use an init container to inject tokens into the config file.

### Upgrade

```bash
helm upgrade llm-proxy ./deployments/helm/llm-proxy -f my-values.yaml
```

### Uninstall

```bash
helm uninstall llm-proxy
```

## Health Checks

The proxy exposes `/healthz` on the configured port. Both liveness and readiness probes in the Helm chart point to this endpoint.

```bash
curl http://localhost:8080/healthz
# ok
```

## Resource Requirements

Default resource requests/limits (configurable via `values.yaml`):

| | CPU | Memory |
|---|---|---|
| Request | 100m | 64Mi |
| Limit | 500m | 128Mi |
