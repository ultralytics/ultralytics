---
comments: true
description: Learn how to install and use Ultralytics AI coding agent skills to enhance your development workflow with intelligent assistance for YOLO projects.
keywords: AI agents, coding agents, GitHub Copilot, Claude Code, OpenAI Codex, VS Code, Cursor, development tools, AI assistance, machine learning workflow, YOLO, Ultralytics, agent skills, developer productivity
---

# AI Coding Agent Skills for Ultralytics

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ai-coding-agents.avif" alt="AI coding agents assisting with Ultralytics YOLO development">
</p>

Developing [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications with Ultralytics YOLO involves numerous tasks, from dataset preparation and model training to export and deployment. [AI coding agents](https://www.ultralytics.com/glossary/ai-agent) can significantly accelerate these workflows by providing intelligent, context-aware assistance. However, generic AI assistants may lack the specialized knowledge required for Ultralytics specific tasks.

Ultralytics provides curated **AI coding agent skills** that equip your AI assistant with expert-level knowledge about YOLO workflows, best practices, and common development patterns. By installing these skills, you can enhance the capabilities of AI coding agents like GitHub Copilot, Claude Code, Cursor, and others, enabling them to provide more accurate and helpful guidance for your Ultralytics projects. [Get started](../quickstart.md) by installing Ultralytics!

## What are AI Coding Agent Skills?

[Agent skills](https://agentskills.io) are structured knowledge modules that provide AI assistants with specialized expertise in specific domains. Think of them as instruction manuals that teach AI agents how to help you with particular tasks. When an AI coding agent has access to Ultralytics skills, it can better understand your YOLO development context and provide more relevant suggestions, code examples, and troubleshooting advice.

Skills are particularly valuable for:

- **Complex workflows**: Multi-step processes like dataset preparation or model training
- **Domain-specific knowledge**: Understanding YOLO specific concepts, arguments, and best practices
- **Consistency**: Ensuring AI suggestions align with Ultralytics conventions and recommended patterns
- **Efficiency**: Reducing the time spent explaining context or correcting AI-generated code

## Available Ultralytics Skills

Ultralytics provides specialized skills covering the most common development tasks:

- **ultralytics-train-model**: Comprehensive guidance for [training](../modes/train.md) YOLO models, including [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning), dataset configuration, and monitoring training progress
- **ultralytics-prepare-dataset**: Best practices for dataset organization, annotation formats, directory structures, and [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) strategies
- **ultralytics-contribute-code**: Guidelines for contributing to the Ultralytics repository, including code standards, PR workflows, and testing requirements
- **ultralytics-troubleshooting**: Common issues, error messages, and debugging strategies for Ultralytics development
- **ultralytics-run-inference**: _Coming soon_ - Instructions for running [inference](../modes/predict.md) with YOLO models on images, videos, streams, and other sources
- **ultralytics-export-model**: _Coming soon_ - Complete [export](../modes/export.md) workflows for converting models to [ONNX](onnx.md), [TensorRT](tensorrt.md), [CoreML](coreml.md), and other formats with optimization recommendations
- **ultralytics-create-custom-model**: _Coming soon_ - Step-by-step guidance for developing custom YOLO architectures and model configurations

Each skill contains detailed workflows, prerequisites, code examples in both Python and CLI formats, and references to relevant documentation pages.

## Installing Skills

Skills can be installed using the Ultralytics CLI with a single command. The installation process copies skill files to a location where your AI coding agent can access them.

!!! info ""

    === "Local/directory (default) install"

        By default, skills are installed to the `.agents/skills/` directory in your current working directory:
        ```bash
        yolo install-skills
        ```
        This creates a local skills directory that AI agents can reference for project specific assistance. Local installation is ideal when working on a specific Ultralytics project and you want skills available only within that project context.

    === "Global (user) install"

        For agents that support global skills (such as GitHub Copilot), you can install skills globally so they're available across all your projects:
        ```bash
        yolo install-skills global agent=copilot
        ```
        Global installation places skills in the standard location for your AI agent, making them accessible system-wide. This is useful when you frequently work with Ultralytics across multiple projects and want consistent AI assistance everywhere.
        The global installation will place the skills in the appropriate directory under the logged-in user's OS home directory. On macOS or Linux, `~/` is used, and for Windows, `%USERPROFILE%` is used, which is equivalent to `C:\Users\{USERNAME}\`.

    === "Custom Directory Installation"

        If you need to install skills to a _specific_ location, you can specify a custom directory (do not include `/skills`, this is automatically appended):
        ```bash
        yolo install-skills dir=/path/to/custom
        ```
        This is useful when working with AI agents that use non-standard skills directories or when you want to manage skills in a centralized location for multiple projects.

        ??? question "Why does using a Custom Directory ignore other arguments?"

            When including a custom directory, the `global` and `agent` arguments are **ignored**. This is because you are specifying _precisely_ where you want the `skills/` directory to reside. Unless you've configured your code Agent to search this directory, the Ultralytics Skills may not be automatically discovered.

### Uninstalling Skills

To remove installed Ultralytics skills (other installed skills remain unaffected), append `uninstall` to the command:

!!! example ""

    === "Local/directory (default) uninstall"

        Removes skills from the default `.agents/skills/` directory in your current working directory:
        ```bash
        yolo install-skills uninstall
        ```

    === "Agent-specific uninstall"

        If you installed skills for a specific agent using the `agent=` argument, include it when uninstalling to target the correct directory:
        ```bash
        yolo install-skills agent=copilot uninstall
        ```

        !!! info ""

            Replace `agent=copilot` with the appropriate agent name.

    === "Global uninstall"

        Removes globally installed skills for the specified agent:
        ```bash
        yolo install-skills global agent=copilot uninstall
        ```

    === "Custom directory uninstall"

        If you installed to a custom location using the `dir=` argument, specify _that_ parent directory (without the ending `skills/`):
        ```bash
        yolo install-skills dir=/path/to/custom/ uninstall
        ```

## Compatible AI Coding Agents

Ultralytics skills follow the [Agent Skills specification](https://agentskills.io), making them compatible with any AI coding agent that supports this standard. Here are some popular agents known to work well with Ultralytics skills:

### VS Code GitHub Copilot

[VS Code GitHub Copilot](https://code.visualstudio.com/docs/copilot/copilot-coding-agent) is an AI pair programmer that suggests code completions and entire functions in real-time within your editor. When configured with Ultralytics skills, Copilot can provide YOLO specific code suggestions that align with best practices. See the VS Code documentation to learn more about [GitHub Copilot Agent Skills in VS Code](https://code.visualstudio.com/docs/copilot/customization/agent-skills).

**Installation**: Use global installation for best results:

!!! example ""

    === "VS Code Copilot Local Skills Install"

        ```bash
        yolo install-skills agent=copilot
        ```

    === "VS Code Global Skills Install"

        ```bash
        yolo install-skills global agent=copilot
        ```

### Cursor

[Cursor](https://www.cursor.com/) is an AI-first code editor built on VS Code that provides intelligent code completion, chat-based assistance, and codebase understanding. Cursor automatically detects skills in the `.agents/skills/` directory. Reference the [Cursor documentation on skills](https://cursor.com/docs/context/skills) for additional details on how skills are handled.

**Installation**: Cursor checks for skills in `.agents/skills`, supporting both local and global installations:

!!! example ""

    === "Cursor Local Skills Install"

        ```bash
        yolo install-skills
        ```

    === "Cursor Global Skills Install"

        ```bash
        yolo install-skills global
        ```

### Claude Code (via Anthropic)

[Claude Code](https://www.anthropic.com/claude) started as a terminal user interface (TUI), with a powerful agent-focused harness for software development. Claude Code will automatically use skills found in a `.claude/skills` directory. Additional information can be found in the [Claude Code documentation on skills](https://code.claude.com/docs/en/skills) on how Claude Code uses skills.

**Installation**: Claude Code checks for skills in `.claude/skills`, supporting both local and global installations:

!!! example ""

    === "Claude Code Local Skills Install"

        ```bash
        yolo install-skills agent=claude
        ```

    === "Claude Code Global Skills Install"

        ```bash
        yolo install-skills global agent=claude
        ```

### OpenAI Codex

[OpenAI's Codex](https://openai.com/blog/openai-codex) and GPT models can be integrated into various development tools and IDEs. [Codex can access skills](https://developers.openai.com/codex/skills) as context when using these models through APIs or custom integrations.

**Installation**: Codex checks for skills in `.agent/skills`, supporting both local and global installations:

!!! example ""

    === "OpenAI Codex Local Skills Install"

        ```bash
        yolo install-skills
        ```

    === "OpenAI Codex Global Skills Install"

        ```bash
        yolo install-skills global
        ```

### Other Compatible Agents

The skills format is designed to be agent-agnostic, so any AI coding assistant that can read and interpret markdown-formatted skills should work. This includes emerging AI development tools and custom agent implementations.

## Using Skills Effectively

Once installed, AI coding agents will automatically reference Ultralytics skills when providing assistance. Here are some tips for getting the most value:

### Be Specific in Your Prompts

While skills provide background knowledge, being specific in your requests helps the AI agent select the most relevant skill and provide targeted assistance:

!!! info ""

    **Generic**: "How do I train a model?"

    **Specific**: "I want to train a YOLO26n detection model on a custom dataset specified in 'my-dataset.yaml' for 100 epochs"

    The specific prompt helps the AI agent leverage the `ultralytics-train-model` skill more effectively.

### Reference Skills Explicitly

Many AI agents allow you to explicitly reference skills in your prompts. This can be particularly useful when working on tasks that span multiple skills:

!!! info ""

    "Using the ultralytics-export-model skill, show me how to export my trained model to TensorRT format with INT8 quantization"

### Combine Skills with Documentation

Skills complement the official Ultralytics documentation rather than replacing it. For best results, skills provide workflow guidance while documentation offers comprehensive API references:

- **Skills**: Best for workflows, examples, and common patterns
- **[Documentation](https://docs.ultralytics.com/)**: Best for detailed API references, advanced configurations, and edge cases

### Keep Skills Updated

As Ultralytics evolves, skills may be updated to reflect new features, best practices, and recommended patterns. Periodically reinstalling skills ensures you have the latest guidance:

!!! example "Upgrading Ultralytics Skills"

    === "Using `uv`"

        ```bash
        uv pip install ultralytics --upgrade && yolo install-skills
        ```

    === "Using `uv` project"

        ```bash
        uv add ultralytics --upgrade && yolo install-skills
        ```

    === "Using `pip`"

        ```bash
        pip install ultralytics --upgrade && yolo install-skills
        ```

    ??? note "Include Arguments when Upgrading"

        The command will upgrade to the latest Ultralytics Python package and agent skills, and without arguments, installs to `.agents/skills` locally. Include `global`, `dir`, and/or `agent` arguments as appropriate for your system.

!!! warning "Upgrading Overwrites Modified Ultralytics Skills"

    If you decide to modify the base Ultralytics agent skills, upgrading will overwrite any custom changes you've made. You should change the name or make a copy of any modified skills _before_ upgrading.

## Skills vs. Other Developer Tools

Ultralytics offers several tools to enhance developer productivity. Here's how skills compare to other options:

| Tool                                     | Purpose                                          | Best For                                                     | Installation                                                                                                |
| ---------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **AI Coding Agent Skills**               | Provide AI assistants with Ultralytics expertise | Enhancing AI agent capabilities across all development tasks | `yolo install-skills`                                                                                       |
| **[VS Code Extension](vscode.md)**       | Code snippets and examples within VS Code        | Accelerating code writing with pre-built snippets            | [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets) |
| **[CLI](../usage/cli.md)**               | Command-line interface for YOLO operations       | Running training, inference, and export from terminal        | Included with `ultralytics` package                                                                         |
| **[Python Package](../usage/python.md)** | Programmatic access to YOLO models               | Building custom applications and scripts                     | `pip install ultralytics`                                                                                   |

These tools are complementary, and you can use skills alongside the VS Code extension, CLI, and Python package to create a comprehensive development environment optimized for Ultralytics workflows.

## Summary

Ultralytics AI coding agent skills bring expert-level YOLO knowledge directly into your development environment, helping AI assistants provide more accurate and context-aware guidance for [model training](../modes/train.md), [inference](../modes/predict.md), [export](../modes/export.md), dataset preparation, and more. With simple CLI installation and support for popular agents like GitHub Copilot, Cursor, Claude Code, and OpenAI Codex, these skills integrate seamlessly into existing workflows. To explore other ways to integrate Ultralytics into your development pipeline, visit the [Integrations guide](index.md).

## FAQ

### What exactly gets installed when I run `yolo install-skills`?

The command copies Markdown files (one for each skill) from the Ultralytics package to the specified skills directory. Each file contains structured information about a specific development task, including workflows, code examples, best practices, and references. These files are read by AI coding agents to enhance their understanding of Ultralytics specific tasks. No executable code is installed; the skills are purely informational Markdown files for AI agents.

### Do I need to install skills separately for each project?

It depends on your preference and AI agent. Local installation (default `yolo install-skills`) places skills in the current project directory, making them project specific. Global installation (`yolo install-skills global agent=copilot`) makes skills available system-wide for supported agents. If you work on multiple Ultralytics projects, global installation may be more convenient. However, local installation gives you control over which projects have access to skills.

### Will skills slow down my AI coding agent?

No. Skills are lightweight Markdown documents that AI agents reference when providing assistance. They don't execute code or run background processes. The AI agent reads skills only when relevant to your current task, similar to how it might reference your project's documentation. The benefits of more accurate, context-aware suggestions typically far outweigh any minimal processing overhead.

### Can I modify the installed skills for my specific use cases?

Yes! After installation, skill files are standard Markdown documents in your skills directory. You can edit them to add project specific information, custom workflows, or internal conventions. However, keep in mind that reinstalling/upgrading skills will overwrite your modifications. If you make significant customizations, consider saving them separately or using a custom directory to avoid accidental overwrites.

### My AI agent doesn't seem to use the skills. What should I try?

First, verify skills are installed in the correct location for your agent. For local installation, check that `.agents/skills/` exists in your project directory. For global installation, check the agent specific directory (e.g., `~/.github/copilot/skills/` for GitHub Copilot). Next, try explicitly referencing a skill in your prompt, such as "Using the ultralytics-train-model skill, show me..." If issues persist, consult your AI agent's documentation for skills support, as not all agents may have enabled this feature yet.

### How do Ultralytics skills compare to fine-tuning an AI model?

Skills and fine-tuning serve different purposes. Fine-tuning permanently modifies an AI model's weights through training on specific data, a process typically controlled by the model provider, not end users. Skills, in contrast, provide runtime context that guides the AI agent's responses without modifying the underlying model. Skills are easier to install, update, and customize, making them more practical for most developers. They're particularly effective for domain-specific knowledge like Ultralytics workflows that may not be well-represented in the AI model's training data.

### Are skills compatible with AI agents other than those listed?

Yes! Skills follow the open [Agent Skills specification](https://agentskills.io), which is designed to be agent-agnostic. Any AI coding assistant that can read and interpret Markdown-formatted skills should work. The specification is relatively new, so support is growing across different AI development tools. To explore other ways to enhance your Ultralytics workflow, check out the full list of [Ultralytics integrations](index.md). If your preferred agent doesn't currently support skills, you can still manually provide skill content as context in your conversations with the AI agent, though this won't be as seamless as native skills support.
