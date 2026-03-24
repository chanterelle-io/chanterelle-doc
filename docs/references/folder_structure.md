# Folder Structure

Chanterelle supports three project types, each identified by a specific JSON file in the project directory. All projects live inside your configured **projects directory**.

## Model Project

A model project is identified by the presence of `model_meta.json`:

```plaintext
my_model_project/
    model_meta.json              # Required — model metadata and UI config
    handler_io.py                # Required — Python handler functions
    model_findings.json          # Optional — static insights/findings
    feedback.jsonl               # Auto-generated — user feedback history
```

➡️ [model_meta.json reference](./model_meta.md) · [handler_io.py reference](./handler_io.md) · [model_findings.json reference](./model_findings.md)

## Analytics Project

An analytics project is identified by the presence of `analytics.json`. No Python code is required — it's purely a JSON-driven dashboard:

```plaintext
my_analytics_project/
    analytics.json               # Required — analytics content and visualizations
    graphs/                      # Optional — images, Plotly JSON, HTML files
        chart.png
        scatter.json
```

➡️ [analytics.json reference](./analytics.md)

## Interactive Project

An interactive project is identified by the presence of `interactive.json`:

```plaintext
my_interactive_project/
    interactive.json             # Required — agent metadata and config
    handler_io.py                # Required — Python handler with initialize() and on_input()
```

➡️ [interactive.json reference](./interactive.md)

## Project Detection

Chanterelle scans the projects directory and detects the project type by looking for these files (in order of priority):

1. `model_meta.json` → **Model** project
2. `interactive.json` → **Interactive** project
3. `analytics.json` → **Analytics** project
