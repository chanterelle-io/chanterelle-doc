---
sidebar_position: 2
---

# AI-Assisted Setup

Chanterelle ships with a built-in **AI skill** that teaches AI coding assistants — GitHub Copilot, Cursor, Windsurf, and others — how to scaffold and configure Chanterelle projects, including `model_meta.json`, `interactive.json`, `analytics.json`, and `handler_io.py`.

This means you can describe what you want in natural language, and your AI assistant will generate valid, well-structured project files for you.

## How It Works

The Chanterelle repository includes a [`chanterelle-project`](https://github.com/chanterelle-io/chanterelle/tree/main/.github/skills/chanterelle-project) skill folder under `.github/skills/`. It contains:

- **`SKILL.md`** — the main entry point that AI assistants read, with the scaffolding procedure and key constraints
- **`references/model-project.md`** — full `model_meta.json` schema and `handler_io.py` function signatures
- **`references/interactive-project.md`** — `interactive.json` schema and interactive `handler_io.py` pattern
- **`references/analytics-project.md`** — `analytics.json` schema
- **`references/visualization-types.md`** — all supported visualization items (tables, charts, Plotly, Markdown, images, etc.)

When the skill is in your workspace, AI assistants automatically pick it up and use it to:

- **Scaffold a new project** — create the right folder structure and files for model, analytics, or interactive projects
- **Write `model_meta.json`** — define inputs, outputs, presets, and UI groupings with correct types and structure
- **Write `interactive.json`** — configure agent metadata and Python environment settings
- **Write `analytics.json`** — build sections with tables, charts, Plotly visualizations, Markdown, and more
- **Generate `handler_io.py`** — produce the correct function signatures (`model_fn`, `predict_fn`, `initialize`, `on_input`, etc.) with proper return formats

## Supported Editors

Any AI coding assistant that reads project-level instruction files:

- **VS Code** with GitHub Copilot
- **Cursor**
- **Windsurf**
- Other AI-enabled editors and IDE plugins

## Quick Start

1. [**Download the skill folder**](/chanterelle-project-skill.zip) and unzip it into your workspace so the structure is `.github/skills/chanterelle-project/`
2. Open your project folder in your editor with your AI assistant enabled
3. Ask the assistant to create a project:

**Example prompts:**

> "Create a Chanterelle model project for an iris classification model with 4 numeric inputs and a category output"

> "Scaffold an interactive project with a conversational SQL agent"

> "Build an analytics dashboard with a bar chart and a summary table"

The assistant will generate the correct files with valid schemas, ready to open in Chanterelle.

## What It Covers

| Project Type | Files Generated | What the AI Handles |
|---|---|---|
| 🧠 **Model** | `model_meta.json`, `handler_io.py`, `model_findings.json` | Input/output definitions, Python function signatures, visualization sections |
| 📊 **Analytics** | `analytics.json` | Sections, items, chart configs, Markdown content, `$href` references |
| 💬 **Interactive** | `interactive.json`, `handler_io.py` | Agent metadata, `initialize()` and `on_input()` functions, dynamic form returns |

## Tips

- The AI knows all 9 input types (`float`, `int`, `string`, `category`, `boolean`, `textarea`, `file`, `button`, `yes_no`) and will pick the right one based on your description
- It understands the visualization system — ask for specific chart types, tables, or Plotly specs and it will produce correct item formats
- For interactive projects, it knows that module-level Python variables persist across turns (the process stays alive)
- You can iterate — ask the assistant to modify or extend generated files, and it will respect the existing structure
