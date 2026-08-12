---
comments: true
description: Install Ultralytics Agent Skills to teach AI coding agents like Claude Code, Codex, Cursor, and Gemini CLI the complete YOLO computer vision workflow, from datasets and training to inference and export.
keywords: Ultralytics, YOLO, agent skills, AI coding agents, Claude Code, Codex, Cursor, Gemini CLI, GitHub Copilot, SKILL.md, computer vision, model training, inference, export, developer productivity
---

# Ultralytics Agent Skills

Agent skills are folders of Markdown instructions and reference material that AI coding agents discover and load on demand, following the open [Agent Skills](https://agentskills.io/) specification. The [ultralytics/skills](https://github.com/ultralytics/skills) repository provides official skills that teach agents like [Claude Code](https://www.claude.com/product/claude-code), [Codex](https://openai.com/codex/), [Cursor](https://cursor.com/), and [Gemini CLI](https://github.com/google-gemini/gemini-cli) how to work with the [Ultralytics Python package](../quickstart.md), the `yolo` CLI, and the [Ultralytics Platform](https://platform.ultralytics.com/).

With the skills installed, your agent answers with accurate, version-grounded Ultralytics knowledge instead of guessing: exact model weight names, valid [train](../modes/train.md) and [predict](../modes/predict.md) arguments, dataset formats, and [export](../modes/export.md) targets.

## Available Skills

The repository contains seven complementary skills covering the full computer vision lifecycle:

| Skill            | Use for                                                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo`           | Core CLI and Python grammar, plus routing to the right lifecycle skill for any Ultralytics task                                                      |
| `yolo-models`    | Choosing a [model](../models/index.md) family (YOLO26, YOLO11, YOLOv8, YOLO-World, YOLOE, SAM, RT-DETR), size, and task variant                      |
| `yolo-datasets`  | Building, converting, validating, and debugging [datasets](../datasets/index.md) — `data.yaml`, label formats, and COCO/DOTA conversion              |
| `yolo-training`  | [Training](../modes/train.md) and fine-tuning workflows, arguments, multi-GPU, resumes, and fixing OOM, NaN loss, or low mAP                         |
| `yolo-tuning`    | [Hyperparameter tuning](../guides/hyperparameter-tuning.md), experiment comparison, and systematic model improvement                                 |
| `yolo-inference` | [Prediction](../modes/predict.md), the Results API, [tracking](../modes/track.md), and [Solutions](../solutions/index.md) like counting and heatmaps |
| `yolo-export`    | [Exporting](../modes/export.md) to ONNX, TensorRT, CoreML, OpenVINO, and NPU formats, quantization, and [benchmarking](../modes/benchmark.md)        |

Each skill ships a `SKILL.md` with decision tables and procedures, plus reference catalogs grounded against a specific `ultralytics` release, so agents stay aligned with real package behavior.

## Installation

!!! example "Install Ultralytics Agent Skills"

    === "Claude Code"

        Install the `yolo` plugin, which bundles all seven skills, from the Ultralytics marketplace:

        ```bash
        claude plugin marketplace add ultralytics/skills
        claude plugin install yolo@ultralytics
        ```

    === "Codex"

        Install the `yolo` plugin, which bundles all seven skills, from the Ultralytics marketplace:

        ```bash
        codex plugin marketplace add ultralytics/skills
        codex plugin add yolo@ultralytics
        ```

    === "Other agents"

        Use the [skills CLI](https://skills.sh/) for Cursor, Gemini CLI, GitHub Copilot, Windsurf, and 70+ other agents. It auto-detects the agents installed on your machine:

        ```bash
        # Install all skills into the current project
        npx skills add ultralytics/skills

        # Install a single skill
        npx skills add ultralytics/skills --skill yolo-training

        # Install globally for all projects
        npx skills add ultralytics/skills -g
        ```

## Usage

Once installed, skills activate automatically whenever a task involves Ultralytics — no extra configuration needed for supported agents' default setups (if your agent was already running when you installed, reload with `/reload-plugins` in Claude Code or restart it). Just ask your agent naturally:

| Category  | Example prompt                                                                |
| --------- | ----------------------------------------------------------------------------- |
| Models    | "Which YOLO26 size should I use for real-time detection on a Jetson?"         |
| Datasets  | "Convert my COCO annotations to YOLO format and build the data.yaml"          |
| Training  | "Fine-tune yolo26s.pt on my dataset for 100 epochs and fix any OOM errors"    |
| Tuning    | "Compare my last two training runs and suggest what to change next"           |
| Inference | "Run tracking on this video and count objects crossing a line"                |
| Export    | "Export my trained model to TensorRT with INT8 quantization and benchmark it" |

The `yolo` skill acts as a router: it matches any Ultralytics task and directs the agent to the lifecycle skill for the stage at hand (training, export, etc.), so deeper references load only when a task requires them and context usage stays minimal.

## FAQ

### What are Ultralytics Agent Skills?

They are official instruction packages from the [ultralytics/skills](https://github.com/ultralytics/skills) repository that follow the open Agent Skills specification. Each skill is a folder with a `SKILL.md` and reference files that AI coding agents load on demand to work accurately with Ultralytics YOLO models, datasets, training, inference, and export.

### Which AI coding agents are supported?

Claude Code and Codex install the skills as a plugin from the Ultralytics marketplace. Cursor, Gemini CLI, GitHub Copilot, Windsurf, and 70+ other agents install them with `npx skills add ultralytics/skills`.

### Do I need to install all seven skills?

Installing the full set is recommended since the skills cross-reference each other, and the `yolo` router skill directs the agent to load only what a task needs, keeping context usage low. If you prefer a minimal setup, install individual skills with `npx skills add ultralytics/skills --skill yolo-training`.

### How do I update the skills?

For Claude Code, run `claude plugin update yolo@ultralytics`, then `/reload-plugins` in your session (or restart) to apply it. For Codex, run `codex plugin marketplace upgrade ultralytics` followed by `codex plugin add yolo@ultralytics`, then restart Codex. For the skills CLI, rerun `npx skills add ultralytics/skills` to pull the latest versions. Skills are grounded against a specific `ultralytics` release, so updating keeps agent knowledge aligned with the package version you use.
