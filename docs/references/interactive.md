# Interactive Projects

Interactive projects enable multi-turn conversational sessions between the user and a Python agent. The agent can present dynamic forms that change each turn, stream outputs with rich visualizations, and maintain conversation state across turns.

## Quick Start

Create an `interactive.json` and a `handler_io.py` in a folder inside your projects directory:

**interactive.json:**
```json
{
  "interactive_name": "My Agent",
  "version": "1.0.0",
  "description_short": "A conversational agent demo",
  "description": "Longer description of what this agent does.",
  "tags": {
    "type": "demo"
  },
  "python_environment": {
    "type": "system"
  }
}
```

**handler_io.py:**
```python
def initialize():
    """Called when the session starts. Returns a greeting and the first form."""
    return {
        "outputs": [
            {
                "type": "section",
                "title": "Welcome",
                "items": [
                    {
                        "type": "text",
                        "id": "greeting",
                        "content": [
                            { "type": "paragraph", "text": "Hello! How can I help you?" }
                        ]
                    }
                ]
            }
        ],
        "next_inputs": [
            {
                "name": "message",
                "label": "Your Message",
                "type": "string",
                "constraints": {
                    "placeholder": "Type something..."
                }
            }
        ]
    }

def on_input(data):
    """Called each time the user submits a form. Returns outputs and the next form."""
    message = data.get("message", "")
    return {
        "outputs": [
            {
                "type": "section",
                "title": "Reply",
                "items": [
                    {
                        "type": "text",
                        "id": "reply",
                        "content": [
                            { "type": "paragraph", "text": f"You said: {message}" }
                        ]
                    }
                ]
            }
        ],
        "next_inputs": [
            {
                "name": "message",
                "label": "Your Message",
                "type": "string",
                "constraints": {
                    "placeholder": "Type something..."
                }
            }
        ]
    }
```

## File Structure

```plaintext
my_interactive_project/
    interactive.json             # Required — agent metadata
    handler_io.py                # Required — Python handler
```

## interactive.json Reference

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `interactive_name` | string | ✅ | Name displayed in the catalog and page header |
| `version` | string | ❌ | Version string |
| `description` | string | ❌ | Detailed description |
| `description_short` | string | ❌ | Brief description shown on the catalog card |
| `tags` | object | ❌ | Key-value pairs for categorization |
| `python_environment` | object | ❌ | Python environment configuration (see [model_meta.md](./model_meta.md#environment-setup)) |

The `python_environment` field uses the same format as model projects (`system`, `venv`, `conda`, `virtualenv`).

## handler_io.py

Interactive projects use a different handler interface than model projects. Instead of the `model_fn`/`input_fn`/`predict_fn`/`output_fn` pattern, interactive handlers use two functions:

### `initialize()`

Called when a new session starts (i.e., when the conversation history is empty). Must return a response object with `outputs` and `next_inputs`.

```python
def initialize():
    return {
        "outputs": [...],      # Rich content sections to display
        "next_inputs": [...]   # Form definition for the next turn
    }
```

### `on_input(data)`

Called each time the user submits a form. Receives a dictionary of the user's input values. Must return a response object with `outputs` and `next_inputs`.

```python
def on_input(data):
    # data is a dict of input values, e.g. {"message": "hello", "option": "A"}
    return {
        "outputs": [...],      # Rich content sections to display
        "next_inputs": [...]   # Form definition for the next turn
    }
```

### Response Format

Both functions return the same structure:

| Field | Type | Description |
|-------|------|-------------|
| `outputs` | array | Array of sections/items (same format as [model_findings.md](./model_findings.md)) |
| `next_inputs` | array | Array of input definitions for the next form (same format as [model_meta.md inputs](./model_meta.md#inputs)) |

**`outputs`** uses the same visualization types as model findings: `text`, `table`, `bar_chart`, `line_chart`, `scatter_plot`, `plotly`, `markdown`, `image`, `html`, `error`, and nested `section`.

**`next_inputs`** uses the same input type definitions as model metadata: `string`, `float`, `int`, `category`, `boolean`, `textarea`, `file`, `button`, `yes_no` — with the same constraints.

### Composer Modes

Chanterelle automatically chooses the best input UI based on the `next_inputs` definition:

| Inputs | UI Mode | Description |
|--------|---------|-------------|
| Single `string` or `textarea` input | **Simple text** | A chat-style text box with send button |
| Single `button` or `yes_no` input | **Quick choice** | Inline buttons the user can click |
| Multiple fields or complex types | **Full form** | A multi-field form that opens as a panel |

## Conversation Flow

1. **Session start**: Chanterelle warms up the Python process, then calls `initialize()` to get the first greeting and form
2. **User submits**: The user fills in the form and submits. Chanterelle calls `on_input(data)` with the form values
3. **Agent responds**: The handler returns `outputs` (displayed as rich content) and `next_inputs` (the next form)
4. **Repeat**: Steps 2–3 repeat indefinitely until the user stops or restarts

The conversation is displayed as alternating user/agent turns in a chat-like interface.

## Stateful Conversations

Since the Python process persists across turns, you can maintain state using module-level variables:

```python
conversation_history = []

def initialize():
    conversation_history.clear()
    return {
        "outputs": [{"type": "section", "title": "Ready", "items": [...]}],
        "next_inputs": [{"name": "message", "label": "Message", "type": "string"}]
    }

def on_input(data):
    conversation_history.append(data)
    # Use conversation_history to provide context-aware responses
    return {
        "outputs": [...],
        "next_inputs": [{"name": "message", "label": "Message", "type": "string"}]
    }
```

## Complete Example

A multi-step agent that collects a name, then enters a message loop with optional chart output:

```python
import json

def initialize():
    return {
        "outputs": [
            {
                "type": "section",
                "title": "Welcome",
                "items": [
                    {
                        "type": "text",
                        "id": "msg",
                        "content": [
                            {"type": "paragraph", "text": "Welcome! Please tell me your name."}
                        ]
                    }
                ]
            }
        ],
        "next_inputs": [
            {
                "name": "username",
                "label": "Your Name",
                "type": "string",
                "constraints": {"placeholder": "Alice"}
            }
        ]
    }

def on_input(data):
    if "username" in data:
        name = data["username"]
        return {
            "outputs": [
                {
                    "type": "section",
                    "title": "Greeting",
                    "items": [
                        {
                            "type": "text",
                            "id": "greet",
                            "content": [
                                {"type": "paragraph", "text": f"Nice to meet you, {name}!"}
                            ]
                        }
                    ]
                }
            ],
            "next_inputs": [
                {"name": "message", "label": "Your Message", "type": "textarea"},
                {"name": "show_chart", "label": "Show Analysis?", "type": "boolean", "default": True}
            ]
        }

    message = data.get("message", "")
    show_chart = data.get("show_chart", False)

    items = [
        {
            "type": "text",
            "id": "echo",
            "content": [
                {"type": "paragraph", "text": f"You said: {message}"},
                {"type": "paragraph", "text": f"Character count: {len(message)}"}
            ]
        }
    ]

    if show_chart:
        items.append({
            "type": "plotly",
            "id": "stats",
            "title": "Message Stats",
            "data": [{
                "x": ["Chars", "Words"],
                "y": [len(message), len(message.split())],
                "type": "bar"
            }],
            "layout": {"height": 300}
        })

    return {
        "outputs": [{"type": "section", "title": "Reply", "items": items}],
        "next_inputs": [
            {"name": "message", "label": "Next Message", "type": "textarea"},
            {"name": "show_chart", "label": "Show Chart", "type": "boolean", "default": show_chart}
        ]
    }
```
