---
plans: [free, pro, enterprise]
comments: true
description: Connect Slack to Ultralytics Platform and choose which training, export, and deployment results are posted to your channel.
keywords: Ultralytics Platform, Slack, alerts, notifications, training, model export, deployment, YOLO, computer vision
title: Slack Alerts - Ultralytics Platform
---

# Slack Integration

Connect [Slack](https://slack.com) to [Ultralytics Platform](https://platform.ultralytics.com) to learn when long-running work finishes without keeping Platform open. You choose one Slack channel and the results that matter to your workspace.

You do not need a Slack API key, webhook, or technical setup. Before you start, sign in to Slack and decide which channel should receive Platform alerts. If you use a team workspace in Platform, you must be a workspace admin or owner.

## Connect Slack

1. Open [**Settings > Integrations**](https://platform.ultralytics.com/settings?tab=integrations) and find the **Slack** card.
2. Click **Add to Slack**.
3. Read the short setup summary and click **Continue to Slack**.
4. Choose a channel and click **Allow**. Platform requests permission to post only to that channel.
5. You are done. All six alerts are enabled; to change them, clear the alerts you do not want and click **Save alerts**.

Platform posts a confirmation in the selected channel as soon as the connection succeeds. Workspace admins manage the connection and alert choices for the whole workspace from the [Integrations tab](../account/settings.md#integrations-tab).

!!! info "What Slack Allows"

    Slack lets Platform post messages to the channel you choose. Platform cannot read your Slack messages or post to other channels through this connection.

## Available Alerts

| Alert                 | When it is sent                                                                |
| --------------------- | ------------------------------------------------------------------------------ |
| **Training complete** | A model [finishes training](../train/cloud-training.md#training-job-lifecycle) |
| **Training failed**   | A training run stops with an error                                             |
| **Export complete**   | A [model export](../train/models.md#export-model) is ready                     |
| **Export failed**     | A model export stops with an error                                             |
| **Deployment ready**  | A [deployment](../deploy/endpoints.md#deployment-lifecycle) is ready           |
| **Deployment failed** | A deployment fails to start                                                    |

Each message says what finished and includes a direct link to the related model or deployment in Platform. Failed-job alerts include a short error summary when one is available. Slack delivery does not change the result of the training, export, or deployment; Platform remains the source of truth, and you can review recent events in [Activity](../account/activity.md).

## Change or Disconnect Slack

To change which results are posted, check or uncheck alerts and click **Save alerts**. At least one alert must remain selected.

To use a different channel, click **Disconnect**, then connect Slack again and choose the new channel. Disconnecting stops all Slack alerts immediately and does not affect Platform jobs or resources.

## Troubleshooting

- **The Slack card says an admin must connect it:** ask a Platform workspace admin or owner to complete the connection.
- **Your Slack workspace or channel is missing:** confirm that you are signed in to the correct Slack workspace and that you can add apps to the channel.
- **The connection worked, but alerts stopped:** reconnect Slack to refresh the channel permission. This is usually needed if the app permission was revoked or the channel was removed.
- **A job finished without a Slack message:** check the selected alerts in **Settings > Integrations**, then confirm the job result in [Platform Activity](../account/activity.md). Slack alerts are informational and never control job processing.

Return to the [Platform integrations overview](index.md) to connect data, storage, or On Premise services.
