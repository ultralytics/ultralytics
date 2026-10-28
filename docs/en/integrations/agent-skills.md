---
comments: true
description: Install and use the official Ultralytics Agent Skills in Claude Code, Codex, and other coding agents for YOLO model selection, datasets, training, tuning, inference, export, and Platform CLI automation.
keywords: Ultralytics, YOLO, agent skills, Claude Code, Codex, Cursor, Gemini CLI, SKILL.md, training, inference, export, Platform CLI, ul cloud, Ask AI, AutoTrain
---

# Ultralytics Agent Skills

[Ultralytics Agent Skills](https://github.com/ultralytics/skills) give compatible AI coding agents instructions and reference material for working with the `ultralytics` Python package, the `yolo` CLI, and [Ultralytics Platform](https://platform.ultralytics.com), including automation with `ul cloud`. They follow the open [Agent Skills](https://agentskills.io/) format and load when a relevant task is requested.

## How the Skills Work

The `yolo` router skill maps a request to the lifecycle stage it involves, and each stage skill covers both routes: the fastest path in Ultralytics Platform, then the local Python and CLI workflow. Version-sensitive facts such as weight names, training arguments, and export formats live in catalogs grounded against a pinned `ultralytics` release, and every skill tells the agent to trust the installed package (`yolo checks`, `yolo cfg`, and error messages) when they differ. The `platform-cli` skill instead defers to the installed `ul` CLI help and the live [Platform API](../platform/api/index.md). Platform's own [Ask AI](../platform/train/autotrain.md) uses the same `platform-cli` skill.

## Available Skills

The repository contains one router, six lifecycle skills, and one Platform CLI skill:

| Skill            | Use for                                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo`           | Core Python and CLI usage, Platform workflows, and routing to the other skills                                                        |
| `yolo-models`    | Selecting model families, sizes, tasks, and weights                                                                                   |
| `yolo-datasets`  | Preparing, converting, validating, and troubleshooting [datasets](../datasets/index.md)                                               |
| `yolo-training`  | [Training](../modes/train.md), validation, resumes, multi-GPU use, and troubleshooting                                                |
| `yolo-tuning`    | [Hyperparameter tuning](../guides/hyperparameter-tuning.md) and experiment improvement                                                |
| `yolo-inference` | [Prediction](../modes/predict.md), results, [tracking](../modes/track.md), and Solutions                                              |
| `yolo-export`    | [Export](../modes/export.md), quantization, deployment formats, and [benchmarking](../modes/benchmark.md)                             |
| `platform-cli`   | Automating [Ultralytics Platform](../platform/index.md) with `ul cloud`: resources, uploads, cloud training, exports, and deployments |

## Installation

=== "Claude Code"

    ```bash
    claude plugin marketplace add ultralytics/skills
    claude plugin install yolo@ultralytics
    ```

    To update, run `claude plugin update yolo@ultralytics` and restart Claude Code.

=== "Codex"

    ```bash
    codex plugin marketplace add ultralytics/skills
    codex plugin add yolo@ultralytics
    ```

    Restart Codex after installation. To update, run `codex plugin marketplace upgrade ultralytics`, reinstall the plugin, and restart.

=== "Other agents"

    Install all eight skills with the [skills CLI](https://www.skills.sh/):

    ```bash
    npx skills add ultralytics/skills
    ```

    Use `--skill yolo-training` to install one skill or `-g` for a global installation.

Each version of the plugin is published as a [GitHub release](https://github.com/ultralytics/skills/releases) with release notes.

### Connect the Platform CLI

The `platform-cli` skill uses the `ul` CLI (Python 3.11+), included with `pip install ultralytics` or the standalone SDK (`pip install ultralytics-platform`). Create a [Platform API key](../platform/account/api-keys.md), make it available to your agent, and check the connection:

```bash
export ULTRALYTICS_API_KEY="YOUR_API_KEY" # or: ul login YOUR_API_KEY
ul cloud account summary
```

## Example Prompts

Once installed, ask the agent naturally. The `yolo` router selects the relevant guidance:

| Goal              | Example prompt                                                                          |
| ----------------- | --------------------------------------------------------------------------------------- |
| Try YOLO          | "Run YOLO on a sample image and show me what it detects."                               |
| Train             | "Turn my images into a custom YOLO model, labeled and trained in Ultralytics Platform." |
| Improve           | "My model's mAP plateaued. Diagnose the run and propose the next experiment."           |
| Deploy            | "Export my trained YOLO model to TensorRT with INT8 quantization and benchmark it."     |
| Analyze video     | "Count the cars crossing a line in traffic.mp4 and save the annotated video."           |
| Automate Platform | "List my Ultralytics Platform datasets and start cloud training on the PPE one."        |

See the [`ultralytics/skills` repository](https://github.com/ultralytics/skills) for current installation commands, source files, updates, and issue reporting.

## FAQ

### What are Ultralytics Agent Skills?

They are reusable instructions and reference material that help AI coding agents work with Ultralytics Platform, the `ultralytics` Python package, the `yolo` CLI, and the `ul` Platform CLI.

### Which agents can use these skills?

The repository provides plugins for Claude Code and Codex. Cursor, Gemini CLI, and other agents that read the open [Agent Skills](https://agentskills.io/) format can install the skill folders directly or through the skills CLI.

### Which skill should I install?

Install the complete plugin for automatic routing across the computer vision lifecycle and Platform CLI operations. To install one skill with the skills CLI, pass its name with `--skill`, such as `--skill yolo-training` or `--skill platform-cli`.

### Do the skills run code or send data?

The plugin itself contains only instructions, so nothing runs when it loads. Commands your agent runs from the skills can download packages, weights, and datasets, and `ul cloud` commands upload the files you pass them to Ultralytics Platform with your API key. The `ultralytics` package also sends anonymous usage analytics, which `yolo settings sync=False` turns off.

### Do the skills replace the Ultralytics documentation?

No. The skills guide an agent toward the relevant workflow, but the installed package and current Ultralytics documentation remain authoritative when behavior or supported options differ.

### How do I report incorrect guidance?

Open a [bug report](https://github.com/ultralytics/skills/issues/new/choose) in the `ultralytics/skills` repository with the prompt you used, the guidance the agent gave, and what should have happened.
