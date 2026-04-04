---
sidebar_position: 5
---

# AI-Assisted Setup

Chanterelle ships with a built-in **AI skill** that teaches AI coding assistants — GitHub Copilot, Cursor, Windsurf, and others — how to scaffold and configure Chanterelle projects, including `model_meta.json`, `interactive.json`, `analytics.json`, and `handler_io.py`.

This means you can describe what you want in natural language, and your AI assistant will generate valid, well-structured project files for you.

## Quick Start

**1. Add the skill to your workspace**

Download the skill folder from the Chanterelle GitHub repository:

👉 [chanterelle-project skill on GitHub](https://github.com/chanterelle-io/chanterelle/tree/main/.github/skills/chanterelle-project)

Unzip or clone it so the structure in your workspace is:

```
your-project/
  .github/
    skills/
      chanterelle-project/
        SKILL.md
        references/
```

**2. Open your project folder in your AI-enabled editor**

Any editor that reads workspace instruction files will automatically pick up the skill.

**3. Ask the assistant to create a project**

Example prompts:

> "Create a Chanterelle model project for an iris classification model with 4 numeric inputs and a category output"

> "Scaffold an interactive project with a conversational SQL agent"

> "Build an analytics dashboard with a bar chart and a summary table"

The assistant will generate correctly structured files, ready to open in Chanterelle.

---

## How It Works

The skill folder contains:

- **`SKILL.md`** — the main entry point with the scaffolding procedure and key constraints
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
