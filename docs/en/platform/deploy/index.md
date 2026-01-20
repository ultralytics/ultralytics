---
comments: true
description: Learn about model deployment options in Ultralytics Platform including inference testing, dedicated endpoints, and monitoring dashboards.
keywords: Ultralytics Platform, deployment, inference, endpoints, monitoring, YOLO, production, cloud deployment
---

# Deployment

[Ultralytics Platform](https://platform.ultralytics.com) provides comprehensive deployment options for putting your YOLO models into production. Test models with the inference API, deploy to dedicated endpoints, and monitor performance in real-time.

## Overview

The Deployment section helps you:

- **Test** models directly in the browser
- **Deploy** to dedicated endpoints in 43 global regions
- **Monitor** request metrics and logs
- **Scale** automatically with traffic

<!-- Screenshot: platform-deploy-overview.avif -->

## Deployment Options

Ultralytics Platform offers multiple deployment paths:

| Option                  | Description                       | Best For                |
| ----------------------- | --------------------------------- | ----------------------- |
| **Test Tab**            | Browser-based inference testing   | Development, validation |
| **Shared API**          | Multi-tenant inference service    | Light usage, testing    |
| **Dedicated Endpoints** | Single-tenant production services | Production, low latency |

## Workflow

```mermaid
graph LR
    A[✅ Test] --> B[⚙️ Configure]
    B --> C[🌐 Deploy]
    C --> D[📊 Monitor]

    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#9C27B0,color:#fff
```

| Stage         | Description                         |
| ------------- | ----------------------------------- |
| **Test**      | Validate model with sample images   |
| **Configure** | Select region and scaling options   |
| **Deploy**    | Create dedicated endpoint           |
| **Monitor**   | Track requests, latency, and errors |

## Architecture

### Shared Inference

The shared inference service runs in 3 key regions:

| Region | Location             |
| ------ | -------------------- |
| US     | Iowa, USA            |
| EU     | Belgium, Europe      |
| AP     | Taiwan, Asia-Pacific |

Requests are routed to your data region automatically.

### Dedicated Endpoints

Deploy to 43 regions worldwide:

- **Americas**: 15 regions
- **Europe**: 12 regions
- **Asia Pacific**: 16 regions

Each endpoint is a single-tenant service with:

- Dedicated compute resources
- Auto-scaling (0-N instances)
- Custom URL
- Independent monitoring

## Key Features

### Global Coverage

Deploy close to your users with 43 regions covering:

- North America, South America
- Europe, Middle East, Africa
- Asia Pacific, Oceania

### Auto-Scaling

Endpoints scale automatically:

- **Scale to zero**: No cost when idle
- **Scale up**: Handle traffic spikes
- **Configurable limits**: Set min/max instances

### Low Latency

Dedicated endpoints provide:

- Cold start: ~2-5 seconds
- Warm inference: 50-200ms (model dependent)
- Regional routing for optimal performance

## Quick Links

- [**Inference**](inference.md): Test models in browser
- [**Endpoints**](endpoints.md): Deploy dedicated endpoints
- [**Monitoring**](monitoring.md): Track deployment performance

## FAQ

### What's the difference between shared and dedicated inference?

| Feature     | Shared          | Dedicated      |
| ----------- | --------------- | -------------- |
| **Latency** | Variable        | Consistent     |
| **Cost**    | Pay per request | Pay for uptime |
| **Scale**   | Limited         | Configurable   |
| **Regions** | 3               | 43             |
| **URL**     | Generic         | Custom         |

### How long does deployment take?

Dedicated endpoint deployment typically takes 1-2 minutes:

1. Image pull (~30s)
2. Container start (~30s)
3. Health check (~30s)

### Can I deploy multiple models?

Yes, each model can have multiple endpoints in different regions. There's no limit on total endpoints (subject to your plan).

### What happens when an endpoint is idle?

With scale-to-zero enabled:

- Endpoint scales down after inactivity
- First request triggers cold start
- Subsequent requests are fast

To avoid cold starts, set minimum instances > 0.
