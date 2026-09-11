---
plans: [free, pro, enterprise]
comments: true
title: Agents - Visual Workflows
description: Build visual computer vision workflows in Ultralytics Platform with YOLO, vision-language models, conditions, dataset collection, and Slack alerts.
keywords: Ultralytics Platform, Agents, workflows, YOLO26, VLM, confidence filtering, datasets, Slack
---

# Agents

[Agents](https://platform.ultralytics.com/agents) connects images, models, conditions, and actions in a visual workflow. Use YOLO to decide which images need a closer look, ask a vision-language model to explain them, or collect selected images in a dataset for review.

Switch to your personal workspace using the sidebar workspace selector and enable **Early access** in **Settings > Profile**. Then open **Agents** in the sidebar or go to [platform.ultralytics.com/agents](https://platform.ultralytics.com/agents). Early access is a personal preference, including when you work in a team workspace.

![Ultralytics Platform Agents canvas showing an image connected to YOLO, a condition, a vision-language model, and an output block](https://cdn.ul.run/i/e551fa972ff3eb23ef89a46fa10afc9e.avif)

## Run Your First Workflow

1. Open **Agents**, click **Add block > Templates**, and choose **YOLO → VLM Monitor**.
2. Select the **Input image** block, choose an uploaded dataset, and select an image containing a person. Use an image of up to 4 megapixels for the shared trial.
3. Keep the default YOLO26 nano model. The condition passes when YOLO detects at least one person.
4. Select **Describe the scene**, choose a vision-language model, and enter a prompt. [Add the matching provider key](account/api-keys.md#provider-keys-for-agents) in **Settings > API Keys** if you have not already done so. In a team workspace, ask the workspace owner to add or replace the provider key.
5. Keep **Shared trial · 60 seconds** selected for a small run, then click **Run**.
6. Watch the blocks update and inspect the **Output**. If no person is detected, the condition finishes without running the description branch.

You can run a draft before saving it. Use **Save workflow** to name and save a workflow for later, and save again after making changes. The **Python** button shows the generated workflow code.

## Start with a Template

| Template                         | Workflow                                                  | Useful for                                                      |
| -------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| **YOLO → VLM Monitor**           | Image → YOLO → condition → vision-language model → Output | Describing a scene only when a person is detected               |
| **Collect Uncertain Detections** | Dataset → Deployment → confidence condition → Dataset     | Building a review set from images with uncertain detections     |
| **Capture and Alert**            | Image → YOLO → condition → Dataset and Slack              | Saving an image and notifying a channel from the same condition |

Templates are editable starting points. Select the images, deployment, destination dataset, and integrations for your workspace before running them. These workflows run when you click **Run**; the Monitor template analyzes the selected input image.

![Ultralytics Platform Agents template picker with YOLO to VLM Monitor, Collect Uncertain Detections, and Capture and Alert](https://cdn.ul.run/i/a460b0cc6ba7cd54e8c5c92f1a1a4a15.avif)

## Choose Where to Run

The execution selector next to **Run** chooses the hardware for the workflow.

| Choice                            | Use it for                                          | Limits and pricing                                                                                                                                         |
| --------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Shared trial · 60 seconds**     | Trying a small workflow                             | Up to 60 seconds, 20 total input images of up to 4 megapixels each, and 20 model/provider calls; one local YOLO model, using an official YOLO26 nano model |
| **Existing dedicated deployment** | Larger datasets, longer runs, or other local models | Uses that deployment's existing hardware and pricing, including an eligible free dedicated deployment; no overall workflow duration or image-count cap     |
| **New deployment…**               | Creating dedicated hardware from the builder        | Opens the existing deployment creation and pricing dialog; choose resources and wait until the deployment is ready                                         |

Shared trials allow two submissions per five minutes. Platform runs support up to 20 blocks. Model/provider calls include YOLO, vision-language models, deployment predictions, and Slack actions; branching can increase the number of calls per input image.

Dedicated runs continue in checkpointed requests on the selected deployment. Closing or reloading the browser does not stop a run. Existing deployment limits and provider limits still apply, and language-model usage is billed by the provider associated with your key. See [deployment pricing](deploy/endpoints.md) for hardware options.

A **Deployment** block selects the endpoint used for a prediction step. The execution selector chooses where the workflow itself runs. You can choose the same dedicated deployment for both.

![Ultralytics Platform Agents execution selector showing a shared 60-second trial, existing dedicated deployments, and New deployment](https://cdn.ul.run/i/39c7339f3027a3b50f93b95ea60b2f36.avif)

## Configure Blocks

| Block                                      | Purpose                                                                               |
| ------------------------------------------ | ------------------------------------------------------------------------------------- |
| **Image**                                  | Select one image from an uploaded dataset                                             |
| **Dataset** without an incoming connection | Read images from a selected dataset and split, with a configurable input limit        |
| **YOLO**                                   | Run an official model or a trained model from the workspace                           |
| **Deployment**                             | Predict through an existing ready Platform deployment                                 |
| **LLM**                                    | Send the image and upstream context to the selected language or vision-language model |
| **Gate**                                   | Control downstream execution with a condition or cadence                              |
| **Dataset** with an incoming connection    | Add the incoming image to a destination dataset                                       |
| **Slack**                                  | Send a message to the workspace's connected Slack channel                             |
| **Export**                                 | Export a model using Platform's existing export workflow                              |
| **Output**                                 | Display the upstream result                                                           |

Select a block to edit its settings. Connect an output to another block to pass its result downstream. One output can connect to multiple blocks: for example, connect one condition to both a destination dataset and Slack. Each branch uses the same evaluated condition.

### Filter Detections with a Condition

Use the same **Gate** block for detection counts and confidence filters. Choose the upstream result, any class or a specific class, and a comparison. For example:

- **Person count ≥ 1**: continue when at least one person is detected.
- **Confidence between 0.25 and 0.5**: continue when at least one matching detection falls within that inclusive range.
- **A specific class in a confidence range**: collect uncertain detections for that class only.

Confidence filters examine individual detections. An image with scores of `0.3` and `0.9` passes a `0.25–0.5` range because one detection matches. Images with no detections do not pass a confidence condition. Set the prediction model's confidence cutoff low enough to retain the detections you want to evaluate.

A condition that does not match completes with no downstream output. Its connected actions do not run for that image.

![Ultralytics Platform Agents condition settings selecting any class and an inclusive detection confidence range from 0.25 to 0.5](https://cdn.ul.run/i/820dc21a5445d5159e3f66ed05d061af.avif)

### Collect Images for Review

Open **Collect Uncertain Detections**, choose a source dataset and split, select your prediction deployment, and configure the confidence range. Select an existing destination dataset in the final **Dataset** block. Create a dataset from the [Datasets page](data/datasets.md) first if needed.

The destination receives the original images, unlabeled, in its `train` split. Existing copies are skipped, and normal dataset storage quotas apply. Source labels and model predictions are not copied as annotations. Open the destination in the existing [annotation editor](data/annotation.md) to review and label the collected images.

Use hosted images that your workspace can access and a destination dataset you can edit. Connected cloud-storage and on-premise datasets cannot be used as collection destinations.

### Send Conditional Slack Alerts

[Connect Slack](integrations/slack.md) in **Settings > Integrations**, then connect a condition to a **Slack** block. Enter the message and use `{output}` to include the upstream result. **Capture and Alert** also connects that same condition to a dataset, so a matching image is collected and a message is sent to the connected channel.

Workflow Slack messages are configured in the block. They are separate from the training, export, and deployment notification choices in Settings.

## Follow Progress and Stop a Run

Active blocks show a green spinner. Completed blocks show a green check and green outline; failed blocks show a red outline. Counts show which inputs have reached each block, and downstream blocks remain unexecuted when a condition does not match.

Reloading the page restores the run's progress and selected execution deployment. Completed results remain visible for the matching workflow. Click **Stop** to cancel an active run; cancellation waits for the executing block to reach a cancellation point. Stopping or deleting its execution deployment also cancels the workflow.

![Ultralytics Platform Agents completed image to YOLO to Output workflow with green checks, processed image counts, and detection results](https://cdn.ul.run/i/8146d518bbcc1e9bee36204c7b00e259.avif)

## Troubleshooting

- **Agents is missing from the sidebar:** enable **Early access** in your personal **Settings > Profile**.
- **The run exceeds the shared trial limits:** select a dedicated deployment, including an eligible free deployment, or reduce the image resolution, input count, and model/provider calls.
- **The deployment is preparing or stopped:** wait for it to become ready or start it from [Deployments](deploy/index.md).
- **A provider key is missing:** [connect the selected provider](account/api-keys.md#provider-keys-for-agents) in **Settings > API Keys**. In a team workspace, ask the workspace owner to add or replace the key.
- **No images reach the destination:** inspect the condition and prediction confidence cutoff. Empty detection results never satisfy a confidence range.
- **Collected images have no labels:** collection adds original images for review. Use the existing annotation tools to label them.
