# Analytics Projects

Analytics projects let you create rich, static insight dashboards without writing any Python code. Define your content in an `analytics.json` file and Chanterelle renders it as a navigable report with a table of contents sidebar.

## Quick Start

Create an `analytics.json` file in a folder inside your projects directory:

```json
{
  "analysis_name": "My Analysis",
  "version": "1.0.0",
  "description_short": "A brief description shown in the catalog",
  "content": [
    {
      "type": "section",
      "id": "overview",
      "title": "Overview",
      "items": [
        {
          "type": "text",
          "id": "intro",
          "title": "Summary",
          "content": [
            { "type": "paragraph", "text": "This is my analytics dashboard." }
          ]
        }
      ]
    }
  ]
}
```

## File Structure

```plaintext
my_analytics_project/
    analytics.json               # Required — analytics content
    graphs/                      # Optional — supporting files
        scatter.png
        chart.json
        report.html
```

## analytics.json Reference

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `analysis_name` | string | ✅ | Name displayed in the catalog and page header |
| `version` | string | ❌ | Version string |
| `description_short` | string | ❌ | Brief description shown on the catalog card |
| `content` | array | ✅ | Array of sections (see below) |

### Content Structure

The `content` array contains sections, which follow the same structure as [model_findings.json sections](./model_findings.md#sections). All the same visualization types are supported:

- `bar_chart`, `line_chart`, `scatter_plot` — Charts
- `table` — Data tables
- `text` — Rich text with paragraphs and bullet lists
- `image` — Images from file paths
- `markdown` — Full Markdown rendering with GFM support
- `plotly` — Interactive Plotly.js charts (inline, file, or JSON string)
- `html` — Embedded HTML (inline or from file)
- `error` — Error messages

Sections support all features including:
- `items_per_row` for grid layout
- `dropdown` + `subsections` for filterable content
- `collapsible` / `collapsed` for collapsible sections
- `$href` for referencing external JSON files

➡️ See the [Model Findings reference](./model_findings.md) for full documentation of all visualization types and section options.

## Complete Example

```json
{
  "version": "1.0.0",
  "analysis_name": "Iris Flower Analysis",
  "description_short": "Scatter plots showing sepal and petal dimensions.",
  "content": [
    {
      "type": "section",
      "id": "overview",
      "title": "Dimension Analysis",
      "description": "Analysis of sepal and petal measurements.",
      "items_per_row": 2,
      "items": [
        {
          "type": "image",
          "id": "sepal_scatter",
          "title": "Sepal Length vs Width",
          "file_path": "graphs/sepal_length_vs_width.png"
        },
        {
          "type": "image",
          "title": "Petal Length vs Width",
          "file_path": "graphs/petal_length_vs_width.png"
        },
        {
          "type": "plotly",
          "title": "Interactive Scatter",
          "data": [
            {
              "x": [1, 2, 3],
              "y": [2, 6, 3],
              "type": "scatter",
              "mode": "lines+markers",
              "marker": { "color": "red" }
            }
          ],
          "layout": { "title": "Sepal Scatter" }
        }
      ]
    },
    {
      "type": "section",
      "id": "charts",
      "title": "Charts from Files",
      "items_per_row": 1,
      "items": [
        {
          "$href": "graphs/sepal_data.json"
        },
        {
          "$href": "graphs/petal_data.json"
        }
      ]
    }
  ]
}
```
