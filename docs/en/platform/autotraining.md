---
plans: [free, pro, enterprise]
comments: true
description: Use Ask AI to manage datasets, train and compare models, annotate images, export, and deploy on Ultralytics Platform.
keywords: Ultralytics Platform, autotraining, Ask AI, Platform CLI, ul cloud, agent skills, Claude Code, Codex
---

# Autotraining with Ask AI

**Ask AI turns your requests into actions across Ultralytics Platform.** Find datasets, train and compare models, annotate images, export, and deploy through a conversation.

Sign in to [Ultralytics Platform](https://platform.ultralytics.com) and click **Ask AI**. The AI agent works directly with your datasets and models, automating tasks such as annotation, training, export, and deployment.

To use Platform actions, first create an [Ultralytics API key](account/api-keys.md) in your personal workspace's **Settings > API Keys**. Ask AI uses your existing personal key automatically; you do not need to paste it into the conversation. Include the project or dataset link when working with a team workspace.

![Ultralytics Platform PPE detection dataset with Ask AI reviewing classes, annotations, and training readiness](https://cdn.ul.run/i/94b13a80c2fb15e32ca6044d0ebe1d37.avif)

## Prompts to Try

Open the relevant dataset or project and ask:

| Goal                  | Example prompt                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Start a project       | Create a private project called "Wildlife Lab". Find and clone a wildlife detection dataset, then prepare a baseline training run. |
| Compare resolutions   | Train three models on this dataset at image sizes 640, 800, and 960. Keep all other settings and dataset splits fixed.             |
| Explore a dataset     | Find container ships, shipping containers, and cranes in xView. Is it a good starting point for port logistics?                    |
| Create a baseline     | Train YOLO26n on this dataset and name the run "xview-baseline".                                                                   |
| Compare models        | Compare the completed models. Which performed best, and why?                                                                       |
| Deploy a model        | Deploy the completed model with the best validation mAP in this project. Give me its endpoint URL and status.                      |
| Auto-annotate         | Use my best compatible model to annotate unlabeled images in this dataset. Preserve existing annotations.                          |
| Review classification | Which model has the best classification accuracy? Which breeds remain confusing, and what should we try next?                      |
| Export a model        | Export this model to ONNX and give me the download link when it's ready.                                                           |

![Ultralytics Platform logistics project with Ask AI comparing completed detection models and explaining validation differences](https://cdn.ul.run/i/33d99ff9fa815aa7ac82af74cdef637f.avif)

## Use Your Own Coding Agent

Ask AI uses the [Platform CLI](https://github.com/ultralytics/sdk), `ul cloud`, and its companion [platform-cli skill](https://github.com/ultralytics/skills). You can use the same commands from Claude Code, Codex, or another compatible coding agent.

### Install and Connect

The setup below is for using the Platform CLI with your own coding agent outside Platform.

The `ul` CLI comes with the Ultralytics package on **Python 3.11+**. Install or update it:

```bash
pip install -U ultralytics
```

Create a [Platform API key](account/api-keys.md) and set it in the terminal used by your agent. Check the connection with:

```bash
export ULTRALYTICS_API_KEY="YOUR_API_KEY"
ul cloud account summary
```

### Add the Skill

Follow the [Agent Skills installation guide](../integrations/agent-skills.md#installation) for Claude Code and Codex plugins. For agents supported by the skills CLI, install the Platform skill with:

```bash
npx skills add ultralytics/skills --skill platform-cli
```

Then use prompts like the examples above, including the project or dataset link for context.

### Run Commands Directly

Use your project and model URL slugs in commands:

```bash
ul cloud datasets list
ul cloud models list project=license-plate
ul cloud models training project=license-plate model=baseline
ul cloud training start --help
```

Use `ul cloud --help` to discover commands and `ul cloud <resource> <operation> --help` for their arguments. See the [Platform API reference](api/index.md) for more details.
