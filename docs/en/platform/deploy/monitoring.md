---
comments: true
description: Monitor deployed YOLO models on Ultralytics Platform with real-time metrics, request logs, and performance dashboards.
keywords: Ultralytics Platform, monitoring, metrics, logs, deployment, performance, YOLO, observability
---

# Monitoring

[Ultralytics Platform](https://platform.ultralytics.com) provides comprehensive monitoring for deployed endpoints. Track request metrics, view logs, and analyze performance in real-time.

<!-- Screenshot: platform-monitoring-page.avif -->

## Monitoring Dashboard

Access the global monitoring dashboard from the sidebar:

1. Click **Monitoring** in the sidebar
2. View all deployments at a glance
3. Click individual endpoints for details

### Overview Cards

<!-- Screenshot: platform-monitoring-cards.avif -->

| Metric                 | Description                         |
| ---------------------- | ----------------------------------- |
| **Total Requests**     | Requests across all endpoints (24h) |
| **Active Deployments** | Currently running endpoints         |
| **Error Rate**         | Percentage of failed requests       |
| **Avg Latency**        | Mean response time                  |

### Deployments Table

<!-- Screenshot: platform-monitoring-table.avif -->

View all deployments with key metrics:

| Column        | Description                 |
| ------------- | --------------------------- |
| **Model**     | Model name with link        |
| **Region**    | Deployed region with flag   |
| **Status**    | Running/Stopped indicator   |
| **Requests**  | Request count (24h)         |
| **Latency**   | P50 response time           |
| **Errors**    | Error count (24h)           |
| **Sparkline** | Traffic trend visualization |

!!! tip "Real-Time Updates"

    The dashboard polls every 30 seconds. Click refresh for immediate updates.

## Endpoint Metrics

View detailed metrics for individual endpoints:

1. Navigate to your model's **Deploy** tab
2. Click on an endpoint
3. View the metrics panel

### Available Metrics

<!-- Screenshot: platform-monitoring-metrics.avif -->

| Metric              | Description                | Unit  |
| ------------------- | -------------------------- | ----- |
| **Request Count**   | Total requests over time   | count |
| **Request Latency** | Response time distribution | ms    |
| **Error Rate**      | Failed request percentage  | %     |
| **Instance Count**  | Active container instances | count |
| **CPU Utilization** | Processor usage            | %     |
| **Memory Usage**    | RAM consumption            | MB    |

### Time Ranges

Select time range for metrics:

| Range   | Description             |
| ------- | ----------------------- |
| **1h**  | Last hour               |
| **6h**  | Last 6 hours            |
| **24h** | Last 24 hours (default) |
| **7d**  | Last 7 days             |

### Metric Charts

Interactive charts show:

- **Line graphs** for trends over time
- **Hover** for exact values
- **Zoom** to analyze specific periods

## Logs

View request logs for debugging:

<!-- Screenshot: platform-monitoring-logs.avif -->

### Log Entries

Each log entry shows:

| Field          | Description          |
| -------------- | -------------------- |
| **Timestamp**  | Request time         |
| **Severity**   | INFO, WARNING, ERROR |
| **Message**    | Log content          |
| **Request ID** | Unique identifier    |

### Severity Levels

Filter logs by severity:

| Level       | Color  | Description         |
| ----------- | ------ | ------------------- |
| **INFO**    | Blue   | Normal requests     |
| **WARNING** | Yellow | Non-critical issues |
| **ERROR**   | Red    | Failed requests     |

### Log Filtering

Filter logs to find issues:

1. Select severity level
2. Search by keyword
3. Filter by time range

## Alerts

Set up alerts for endpoint issues (coming soon):

| Alert Type          | Trigger                   |
| ------------------- | ------------------------- |
| **High Error Rate** | Error rate > threshold    |
| **High Latency**    | P95 latency > threshold   |
| **No Requests**     | Zero requests for period  |
| **Scaling**         | Instances at max capacity |

## Performance Optimization

Use monitoring data to optimize:

### High Latency

If latency is too high:

1. Check instance count (may need more)
2. Verify model size is appropriate
3. Consider closer region
4. Check image sizes being sent

### High Error Rate

If errors are occurring:

1. Review error logs for details
2. Check request format
3. Verify API key is valid
4. Check rate limits

### Scaling Issues

If hitting capacity:

1. Increase max instances
2. Set min instances > 0
3. Consider multiple regions
4. Optimize request batching

## Export Data

Export monitoring data for analysis:

1. Select time range
2. Click **Export**
3. Download CSV file

Export includes:

- Timestamp
- Request count
- Latency metrics
- Error counts
- Instance metrics

## FAQ

### How long is data retained?

| Data Type   | Retention |
| ----------- | --------- |
| **Metrics** | 30 days   |
| **Logs**    | 7 days    |
| **Alerts**  | 90 days   |

### Can I set up external monitoring?

Yes, endpoint URLs work with external monitoring tools:

- Uptime monitoring (Pingdom, UptimeRobot)
- APM tools (Datadog, New Relic)
- Custom health checks

### How accurate are the latency numbers?

Latency metrics measure:

- **P50**: Median response time
- **P95**: 95th percentile
- **P99**: 99th percentile

These represent server-side processing time, not including network latency to your users.

### Why are my metrics delayed?

Metrics have a ~2 minute delay due to:

- Metrics aggregation pipeline
- Aggregation windows
- Dashboard caching

For real-time debugging, check logs which are near-instant.

### Can I monitor multiple endpoints together?

Yes, the global monitoring dashboard shows all endpoints. Use the table to compare performance across deployments.
