---
comments: true
description: Install the official Ultralytics Agent Skills for YOLO model selection, datasets, training, tuning, inference, and export.
keywords: Ultralytics, YOLO, agent skills, Claude Code, Codex, Cursor, Gemini CLI, SKILL.md, training, inference, export
---

# Ultralytics Agent Skills

[Ultralytics Agent Skills](https://github.com/ultralytics/skills) give compatible AI coding agents instructions and reference material for working with the `ultralytics` Python package, the `yolo` CLI, and [Ultralytics Platform](https://platform.ultralytics.com/). They follow the open [Agent Skills](https://agentskills.io/) format and load when a relevant task is requested.

## Available Skills

The repository contains one router and six lifecycle skills:

| Skill            | Use for                                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `yolo`           | Core Python and CLI usage, Platform workflows, and routing to the other skills                            |
| `yolo-models`    | Selecting model families, sizes, tasks, and weights                                                       |
| `yolo-datasets`  | Preparing, converting, validating, and troubleshooting [datasets](../datasets/index.md)                   |
| `yolo-training`  | [Training](../modes/train.md), validation, resumes, multi-GPU use, and troubleshooting                    |
| `yolo-tuning`    | [Hyperparameter tuning](../guides/hyperparameter-tuning.md) and experiment improvement                    |
| `yolo-inference` | [Prediction](../modes/predict.md), results, [tracking](../modes/track.md), and Solutions                  |
| `yolo-export`    | [Export](../modes/export.md), quantization, deployment formats, and [benchmarking](../modes/benchmark.md) |

The skills include version-grounded catalogs where exact weights, arguments, or export formats matter, while deferring to the installed package when behavior differs.

## Installation

=== "Claude Code"

    ```bash
    claude plugin marketplace add ultralytics/skills
    claude plugin install yolo@ultralytics
    ```

=== "Codex"

    ```bash
    codex plugin marketplace add ultralytics/skills
    codex plugin add yolo@ultralytics
    ```

    Restart Codex after installation. To update, run `codex plugin marketplace upgrade ultralytics`, reinstall the plugin, and restart.

=== "Other agents"

    Install all seven skills with the [skills CLI](https://skills.sh/):

    ```bash
    npx skills add ultralytics/skills
    ```

    Use `--skill yolo-training` to install one skill or `-g` for a global installation.

Once installed, ask the agent naturally, for example: "Export my trained YOLO model to TensorRT with INT8 quantization and benchmark it." The `yolo` router selects the relevant lifecycle guidance.

See the [`ultralytics/skills` repository](https://github.com/ultralytics/skills) for current installation commands, source files, updates, and issue reporting.
