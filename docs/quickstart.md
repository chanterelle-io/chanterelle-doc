---
sidebar_position: 3
---

# Quickstart

Build and open a minimal **Model project** in Chanterelle in under 5 minutes.

## Prerequisites

- Chanterelle installed and configured ([Installation guide](./install))
- Python 3.8+

---

## Step 1 — Create a project folder

Inside your Chanterelle projects directory, create a new folder:

```
chanterelle-projects/
  my-first-model/        ← create this
```

Chanterelle discovers projects by scanning for `model_meta.json`, `analytics.json`, or `interactive.json` files one level deep in the projects directory.

## Step 2 — Write `model_meta.json`

Create `my-first-model/model_meta.json`:

```json
{
  "model_id": "my-first-model",
  "model_name": "My First Model",
  "model_version": "1.0.0",
  "description_short": "Multiplies two numbers together.",
  "description": "A minimal model that takes two numbers and returns their product.",
  "inputs": [
    {
      "name": "a",
      "label": "Value A",
      "type": "float",
      "required": true,
      "description": "The first number",
      "default": 3.0
    },
    {
      "name": "b",
      "label": "Value B",
      "type": "float",
      "required": true,
      "description": "The second number",
      "default": 4.0
    }
  ],
  "outputs": [
    {
      "name": "product",
      "label": "Product",
      "type": "float",
      "description": "The result of A × B"
    }
  ]
}
```

## Step 3 — Write `handler_io.py`

Create `my-first-model/handler_io.py`:

```python
def model_fn():
    """Load your model. The return value is passed to predict_fn as `model`."""
    return None  # No model file needed for this example


def input_fn(inputs: dict, model):
    """Transform raw inputs. The return value is passed to predict_fn as `data`."""
    return inputs


def predict_fn(data: dict, model):
    """Run inference. The return value is passed to output_fn as `prediction`."""
    return {"product": data["a"] * data["b"]}


def output_fn(prediction: dict) -> dict:
    """Format the prediction for display. Return a dict matching your outputs."""
    return prediction
```

## Step 4 — Open in Chanterelle

1. Open Chanterelle.
2. Your project directory is scanned automatically — **My First Model** should appear in the catalog.
3. Click the project card to open it.
4. You'll see a form with **Value A** and **Value B** fields and a **Run** button.
5. Fill in values and click **Run** — the **Product** output appears below.

That's it.

---

## What's Next?

| Goal | Where to go |
|---|---|
| Add a dropdown input | [`category` input type](./references/model_meta#input-types) |
| Add preset example inputs | [`input_presets`](./references/model_meta#input-presets) |
| Show a chart alongside predictions | [`model_findings.json`](./references/model_findings) |
| No predictions, just a dashboard | [Analytics projects](./references/analytics) |
| Conversational agent | [Interactive projects](./references/interactive) |
| Generate files with AI | [AI-Assisted Setup](./ai-assisted-setup) |
