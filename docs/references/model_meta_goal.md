# Model Metadata Reference

## TODO
### Mentioned in this document
- string constraints: add min_length and max_length
- required in conditional inputs? e.g. Show/hide inputs based on boolean flags
- Complex structured output
- output type | `array` | List/table view | Depends on item type | Multiple predictions, time series
- collapsible input groupings
- Docker environment setup
### Not mentioned in this document
- Add images and small videos under each input/output description to enhance understanding.
- Find better examples

## Overview
The model metadata is the configuration file that defines how your machine learning model integrates with the Chanterelle platform. It defines the structure, inputs, outputs, and UI settings for your model. This JSON-based configuration controls model discovery, user interface generation, input validation, and execution environment setup.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Basic Structure](#basic-structure)
- [Model Information](#model-information)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [UI Configuration](#ui-configuration)
- [Environment Setup](#environment-setup)
- [Complete Examples](#complete-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)

## Quick Start

> **New to Chanterelle?** Start with this minimal example and expand as needed.

```json
{
  "model_id": "my-first-model",
  "model_name": "My First Model",
  "model_version": "1.0.0",
  "description": "A simple example model for demonstration",
  "inputs": [
    {
      "name": "input_value",
      "label": "Input Value",
      "type": "float",
      "required": true,
      "description": "The main input parameter",
      "default": 1.0
    }
  ],
  "outputs": [
    {
      "name": "result",
      "label": "Result",
      "type": "float",
      "description": "The computed result"
    }
  ]
}
```

Save this as `model_meta.json` in your model directory and you're ready to go!

## Overview

The metadata file describes:
- **Model information** (ID, name, version, description)
- **Input parameters** with types, constraints, and validation
- **Output specifications** 
- **UI configuration** including groupings and presets
- **Python environment** requirements

## Basic Structure

A model metadata file follows this JSON schema:

```json
{
  // Core model identification
  "model_id": "unique-model-identifier",
  "model_name": "Human Readable Model Name",
  "model_version": "1.0.0",
  "description": "Detailed description of model capabilities and use cases",
  
  // Optional descriptive fields
  "description_short": "Brief one-line summary",
  "links": [...],           // External documentation
  "tags": {...},            // Categorization metadata
  
  // Model interface definition
  "inputs": [...],          // Input parameter specifications
  "outputs": [...],         // Output format definitions
  
  // UI enhancement (optional)
  "input_presets": [...],   // Predefined parameter combinations
  "input_groupings": [...], // UI organization
  
  // Execution environment (optional)
  "python_environment": {...}
}
```

## Model Information

The model information section contains metadata that identifies your model and provides context to users.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Unique identifier for your model |
| `model_name` | string | Human-readable name displayed in the UI |
| `model_version` | string | Version string (e.g., "1.0.0") |
| `description` | string | Detailed description of the model |
| `inputs` | array | List of input parameters (see [Inputs](#inputs)) |
| `outputs` | array | List of output parameters (see [Outputs](#outputs)) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `description_short` | string | Brief one-line description |
| `links` | array | External links or documentation |
| `tags` | object | Key-value pairs for categorization (see [Tags](#tags)) |
| `input_presets` | array | Predefined input combinations (see [Input Presets](#input-presets)) |
| `input_groupings` | array | UI grouping for input parameters (see [Input Groupings](#input-groupings)) |
| `python_environment` | object | Python environment configuration (see [Python Environment](#python-environment)) |

## Inputs

Input parameters define the interface between users and your model. Each input creates a form field in the Chanterelle UI with appropriate validation and controls.

### Input Fields Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Parameter name (used in code) |
| `label` | string | ✅ | Display name in UI |
| `type` | string | ✅ | Data type (see [Input Types](#input-types)) |
| `description` | string | ✅ | Help text explaining the parameter |
| `required` | boolean | ❌ | Whether input is mandatory (default: `false`) |
| `default` | any | ❌ | Default value |
| `unit` | string | ❌ | Unit of measurement (e.g., "seconds", "percent") |
| `constraints` | object | ❌ | Validation rules (see [Input Constraints](#input-constraints)) |
| `depends_on` | object | ❌ | Conditional behavior (see [Conditional Inputs](#conditional-inputs)) |

### Basic Input Structure

```json
{
  "name": "input_name",
  "label": "Display Name",
  "type": "float",
  "required": true,
  "description": "Description of this input",
  "default": 0.001,
  "unit": "rate",
  "constraints": {
    "min": 0.0001,
    "max": 1.0,
    "step": 0.0001
  }
}
```

### Input Types

Chanterelle supports five core input types, each with specific UI controls and validation:

| Type | UI Control | Description | Example Values |
|------|------------|-------------|----------------|
| `float` | Number input with decimals | Floating-point numbers | `3.14`, `0.5`, `-2.7` |
| `int` | Number input (whole numbers) | Integer values | `1`, `42`, `-10` |
| `string` | Text input | Text values | `"hello world"`, `"model_v2"` |
| `category` | Dropdown/Select | Selection from predefined options | See [constraints](#category-constraints) |
| `boolean` | Checkbox/Toggle | True/false values | `true`, `false` |

### Type-Specific Examples

**Float with validation:**
```json
{
  "name": "dropout_rate",
  "label": "Dropout Rate",
  "type": "float",
  "description": "Probability of dropping neurons during training",
  "default": 0.2,
  "unit": "probability",
  "constraints": {
    "min": 0.0,
    "max": 1.0,
    "step": 0.05
  }
}
```

**Integer with range:**
```json
{
  "name": "num_epochs",
  "label": "Training Epochs",
  "type": "int",
  "description": "Number of complete passes through the training data",
  "default": 100,
  "constraints": {
    "min": 1,
    "max": 1000
  }
}
```

**String with pattern validation:**
```json
{
  "name": "experiment_name",
  "label": "Experiment Name",
  "type": "string",
  "description": "Unique identifier for this training run",
  "default": "experiment_001",
  "constraints": {
    "regex": "^[a-zA-Z0-9_-]+$"
  }
}
```

### Input Constraints

Constraints provide validation and UI enhancement for input parameters. They ensure data integrity and improve user experience.

#### Numeric Constraints (float/int)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `min` | number | Minimum allowed value | `0.001` |
| `max` | number | Maximum allowed value | `1.0` |
| `step` | number | Step size for UI controls | `0.001` |

**Example with validation messages:**
```json
{
  "name": "confidence_threshold",
  "label": "Confidence Threshold",
  "type": "float",
  "description": "Minimum confidence for predictions to be considered valid",
  "default": 0.7,
  "constraints": {
    "min": 0.0,
    "max": 1.0,
    "step": 0.01
  }
}
```

#### Category Constraints

For selection inputs, define available options either with simple strings or rich objects that include labels and descriptions:

**Simple string array for basic selections:**
```json
{
  "name": "device",
  "label": "Compute Device",
  "type": "category",
  "description": "Hardware to use for model execution",
  "default": "auto",
  "constraints": {
    "options": ["auto", "cpu", "gpu", "tpu"]
  }
}
```

**Rich options with descriptions:**
```json
{
  "name": "optimizer",
  "label": "Optimization Algorithm",
  "type": "category",
  "description": "Choose the optimization algorithm for training",
  "default": "adam",
  "constraints": {
    "options": [
      {
        "value": "sgd",
        "label": "Stochastic Gradient Descent",
        "description": "Classic optimization with momentum support"
      },
      {
        "value": "adam",
        "label": "Adam Optimizer",
        "description": "Adaptive learning rate with momentum (recommended)"
      },
      {
        "value": "rmsprop",
        "label": "RMSprop",
        "description": "Good for recurrent neural networks"
      }
    ]
  }
}
```

#### String Constraints

Control string input format and validation:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `regex` | string | Regular expression pattern | `"^[a-zA-Z0-9_-]+$"` |
| `min_length` (not yet implemented) | integer | Minimum string length | `3` |
| `max_length` (not yet implemented) | integer | Maximum string length | `50` |

**Example with multiple constraints:**
```json
{
  "name": "model_checkpoint",
  "label": "Model Checkpoint Path",
  "type": "string",
  "description": "Path to saved model weights (relative to model directory)",
  "default": "checkpoints/best_model.pth",
  "constraints": {
    "regex": "^[a-zA-Z0-9_/-]+\\.(pth|pt|h5|ckpt)$",
    "max_length": 100
  }
}
```

### Conditional Inputs

Create dynamic forms where inputs change based on other input values. This enables sophisticated parameter configurations that adapt to user choices.

#### Basic Conditional Structure

```json
{
  "name": "regularization_strength",
  "label": "Regularization Strength",
  "type": "float",
  "description": "Strength of regularization (algorithm-dependent)",
  "depends_on": {
    "input_name": "algorithm",
    "mapping": {
      "svm": {
        "constraints": {
          "min": 0.001,
          "max": 10.0,
          "step": 0.001
        }
      },
      "logistic": {
        "constraints": {
          "min": 0.01,
          "max": 100.0,
          "step": 0.01
        }
      }
    }
  }
}
```

#### Advanced Conditional Example

Show/hide inputs based on boolean flags:

```json
{
  "name": "custom_loss_weight",
  "label": "Custom Loss Weight",
  "type": "float",
  "description": "Weight for custom loss component",
  "depends_on": {
    "input_name": "use_custom_loss",
    "mapping": {
      "true": {
        "required": true,
        "default": 0.5,
        "constraints": {
          "min": 0.0,
          "max": 1.0,
          "step": 0.1
        }
      },
      "false": {
        "required": false
      }
    }
  }
}
```

#### Multiple Dependencies

Complex conditional logic with multiple parent inputs:

```json
{
  "name": "attention_heads",
  "label": "Number of Attention Heads",
  "type": "int",
  "description": "Number of attention heads (only for transformer models)",
  "depends_on": {
    "input_name": "model_architecture",
    "mapping": {
      "transformer": {
        "required": true,
        "default": 8,
        "constraints": {
          "min": 1,
          "max": 32
        },
        "depends_on": {
          "input_name": "model_size",
          "mapping": {
            "small": {"default": 4, "constraints": {"max": 8}},
            "medium": {"default": 8, "constraints": {"max": 16}},
            "large": {"default": 16, "constraints": {"max": 32}}
          }
        }
      }
    }
  }
}
```

## Outputs

Output definitions specify what your model returns and how results should be displayed. Well-defined outputs enable rich visualizations and proper data interpretation.

### Output Fields Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Output identifier (used in code) |
| `label` | string | ✅ | Display name in UI |
| `type` | string | ✅ | Data type (same as input types) |
| `description` | string | ✅ | Explanation of what this output represents |
| `unit` | string | ❌ | Unit of measurement |
| `min` | number | ❌ | Expected minimum value (for visualization) |
| `max` | number | ❌ | Expected maximum value (for visualization) |
| `options` | array | ❌ | Possible values for categorical outputs |
| `format` | string | ❌ | Display format hint (e.g., "percentage", "currency") |

### Basic Output Examples

**Numeric output with range:**
```json
{
  "name": "confidence_score",
  "label": "Prediction Confidence",
  "type": "float",
  "description": "Model confidence in the prediction (higher is more confident)",
  "unit": "probability",
  "min": 0.0,
  "max": 1.0,
  "format": "percentage"
}
```

**Categorical output with options:**
```json
{
  "name": "predicted_class",
  "label": "Predicted Category",
  "type": "string",
  "description": "The most likely category for the input",
  "options": [
    {
      "value": "positive",
      "label": "Positive Sentiment",
      "description": "Text expresses positive sentiment"
    },
    {
      "value": "negative", 
      "label": "Negative Sentiment",
      "description": "Text expresses negative sentiment"
    },
    {
      "value": "neutral",
      "label": "Neutral Sentiment", 
      "description": "Text is neutral or mixed sentiment"
    }
  ]
}
```

**Complex structured output:**
```json
{
  "name": "detection_results",
  "label": "Object Detections",
  "type": "array",
  "description": "List of detected objects with bounding boxes and confidence scores",
  "item_type": "object",
  "schema": {
    "bbox": {
      "type": "array",
      "description": "Bounding box coordinates [x, y, width, height]"
    },
    "class": {
      "type": "string", 
      "description": "Detected object class"
    },
    "confidence": {
      "type": "float",
      "description": "Detection confidence",
      "min": 0.0,
      "max": 1.0
    }
  }
}
```

### Output Types and Formatting

| Type | UI Display | Format Options | Example |
|------|------------|----------------|---------|
| `float` | Number with decimals | `percentage`, `currency`, `scientific` | Confidence scores, probabilities |
| `int` | Whole number | `count`, `ordinal` | Class indices, rankings |
| `string` | Text display | `code`, `multiline` | Class names, generated text |
| `boolean` | Yes/No badge | N/A | Binary classification results |
| `array` | List/table view | Depends on item type | Multiple predictions, time series |

### Advanced Output Features

**Time series output:**
```json
{
  "name": "forecast",
  "label": "Forecasted Values",
  "type": "array",
  "description": "Predicted values over time",
  "item_type": "object",
  "visualization": "line_chart",
  "schema": {
    "timestamp": {
      "type": "string",
      "format": "datetime"
    },
    "value": {
      "type": "float",
      "unit": "units"
    },
    "confidence_interval": {
      "type": "object",
      "schema": {
        "lower": {"type": "float"},
        "upper": {"type": "float"}
      }
    }
  }
}
```

## UI Configuration

Enhance the user experience with input presets and logical groupings that make complex models more approachable.

### Input Presets

Presets provide predefined parameter combinations for common use cases, helping users get started quickly with proven configurations.

#### Basic Preset Structure

```json
{
  "input_presets": [
    {
      "input_preset": "training_modes",
      "label": "Training Mode",
      "description": "Common training configurations optimized for different scenarios",
      "affects": ["epochs", "batch_size", "learning_rate", "dropout_rate"],
      "presets": [
        {
          "name": "Quick Test",
          "description": "Fast training for initial validation",
          "values": {
            "epochs": 10,
            "batch_size": 64,
            "learning_rate": 0.01,
            "dropout_rate": 0.1
          }
        },
        {
          "name": "Production Training",
          "description": "Thorough training for deployment",
          "values": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "dropout_rate": 0.2
          }
        },
        {
          "name": "Fine-tuning",
          "description": "Gentle training for pre-trained models",
          "values": {
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "dropout_rate": 0.05
          }
        }
      ]
    }
  ]
}
```

#### Conditional Presets

Create presets that adapt based on other input values:

```json
{
  "input_preset": "optimization_preset",
  "label": "Optimization Settings", 
  "description": "Optimizer-specific parameter presets",
  "affects": ["learning_rate", "momentum", "weight_decay"],
  "depends_on": {
    "field": "optimizer",
    "mapping": {
      "sgd": {
        "constraints": {},
        "presets": [
          {
            "name": "Conservative",
            "values": {
              "learning_rate": 0.001,
              "momentum": 0.9,
              "weight_decay": 0.0001
            }
          },
          {
            "name": "Aggressive", 
            "values": {
              "learning_rate": 0.01,
              "momentum": 0.95,
              "weight_decay": 0.001
            }
          }
        ]
      },
      "adam": {
        "constraints": {},
        "presets": [
          {
            "name": "Default",
            "values": {
              "learning_rate": 0.001,
              "weight_decay": 0.0
            }
          },
          {
            "name": "With Regularization",
            "values": {
              "learning_rate": 0.0005,
              "weight_decay": 0.01
            }
          }
        ]
      }
    }
  }
}
```

### Input Groupings

Organize related parameters into logical sections for better UI layout and user comprehension.

```json
{
  "input_groupings": [
    {
      "grouping": "model_architecture",
      "description": "Core model structure and design",
      "inputs": ["model_type", "hidden_layers", "activation_function"],
      "collapsible": false,
      "order": 1
    },
    {
      "grouping": "training_parameters", 
      "description": "Training process configuration",
      "inputs": ["epochs", "batch_size", "learning_rate", "optimizer"],
      "collapsible": true,
      "order": 2
    },
    {
      "grouping": "regularization",
      "description": "Overfitting prevention techniques",
      "inputs": ["dropout_rate", "weight_decay", "early_stopping"],
      "collapsible": true,
      "order": 3
    },
    {
      "grouping": "advanced_options",
      "description": "Expert-level configuration options",
      "inputs": ["gradient_clipping", "custom_loss_weight"],
      "collapsible": true,
      "collapsed_by_default": true,
      "order": 4
    }
  ]
}
```

#### Grouping Options

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `grouping` | string | Unique identifier for the group | Required |
| `description` | string | Help text displayed with the group | Required |
| `inputs` | array | List of input names in this group | Required |
| `collapsible` | boolean | Whether group can be collapsed/expanded | `true` |
| `collapsed_by_default` | boolean | Initial collapsed state | `false` |
| `order` | integer | Display order (lower numbers first) | Auto-assigned |

## Environment Setup

Configure the Python execution environment for your model.

### Python Environment Configuration

```json
{
  "python_environment": {
    "type": "conda",
    "name": "my-model-env",
    "python_version": "3.9",
    "requirements": [
      "torch>=1.12.0",
      "transformers>=4.20.0", 
      "numpy>=1.21.0",
      "scikit-learn>=1.1.0"
    ],
    "channels": ["conda-forge", "pytorch"],
    "pip_requirements": [
      "custom-package==1.2.3"
    ]
  }
}
```

### Environment Types

| Type | Description | Configuration Options |
|------|-------------|----------------------|
| `system` | Use system Python | None |
| `venv` | Python virtual environment | `path`, `requirements` |
| `conda` | Conda environment | `name`, `channels`, `requirements`, `pip_requirements` |
| `virtualenv` | Legacy virtualenv | `path`, `requirements` |
| `docker` | Docker container | `image`, `dockerfile_path` |

### Docker Environment

For complex dependencies or reproducible environments:

```json
{
  "python_environment": {
    "type": "docker",
    "image": "pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime",
    "dockerfile_path": "./docker/Dockerfile",
    "build_args": {
      "PYTHON_VERSION": "3.9"
    },
    "volumes": [
      "./data:/app/data:ro",
      "./models:/app/models"
    ],
    "environment_variables": {
      "CUDA_VISIBLE_DEVICES": "0",
      "TOKENIZERS_PARALLELISM": "false"
    }
  }
}
```

## Links and Documentation

Provide additional resources and documentation references:

```json
{
  "links": [
    {
      "name": "Research Paper",
      "url": "https://arxiv.org/abs/1234.5678",
      "description": "Original paper describing the algorithm"
    },
    {
      "name": "User Manual",
      "file_name": "docs/user_manual.pdf",
      "description": "Detailed usage instructions"
    },
    {
      "name": "Model Card",
      "url": "https://huggingface.co/models/my-model",
      "description": "Detailed model information and performance metrics"
    },
    {
      "name": "API Documentation", 
      "url": "https://docs.example.com/api",
      "description": "Programming interface documentation"
    }
  ]
}
```

## Tags

Categorize and make your model discoverable:

```json
{
  "tags": {
    "domain": "natural_language_processing",
    "task": "sentiment_analysis",
    "framework": "pytorch",
    "language": "english",
    "difficulty": "intermediate",
    "training_data": "social_media",
    "performance": "production_ready",
    "license": "mit"
  }
}
```

### Common Tag Categories

| Category | Example Values | Purpose |
|----------|----------------|---------|
| `domain` | `computer_vision`, `nlp`, `audio`, `tabular` | Primary application area |
| `task` | `classification`, `regression`, `detection`, `generation` | Specific ML task |
| `framework` | `pytorch`, `tensorflow`, `scikit_learn`, `xgboost` | Implementation framework |
| `difficulty` | `beginner`, `intermediate`, `advanced`, `expert` | User skill level |
| `performance` | `research`, `prototype`, `production_ready` | Maturity level |
| `license` | `mit`, `apache`, `gpl`, `proprietary` | Usage rights |

## Complete Examples

Real-world examples demonstrating different types of models and use cases.

### Example 1: Simple Regression Model

A basic model for predicting house prices:

```json
{
  "model_id": "house-price-predictor",
  "model_name": "House Price Predictor",
  "model_version": "1.0.0",
  "description_short": "Linear regression model for real estate price estimation",
  "description": "A scikit-learn based linear regression model trained on housing market data. Predicts house prices based on location, size, and property features.",
  "tags": {
    "domain": "real_estate",
    "task": "regression",
    "framework": "scikit_learn",
    "difficulty": "beginner"
  },
  "inputs": [
    {
      "name": "square_feet",
      "label": "Square Footage",
      "type": "int",
      "required": true,
      "description": "Total living space in square feet",
      "default": 2000,
      "unit": "sq ft",
      "constraints": {
        "min": 500,
        "max": 10000,
        "step": 50
      }
    },
    {
      "name": "bedrooms",
      "label": "Number of Bedrooms",
      "type": "int",
      "required": true,
      "description": "Number of bedrooms in the house",
      "default": 3,
      "constraints": {
        "min": 1,
        "max": 8
      }
    },
    {
      "name": "location",
      "label": "Location",
      "type": "category",
      "required": true,
      "description": "Neighborhood or city district",
      "default": "downtown",
      "constraints": {
        "options": [
          {"value": "downtown", "label": "Downtown", "description": "City center area"},
          {"value": "suburbs", "label": "Suburbs", "description": "Residential suburbs"},
          {"value": "rural", "label": "Rural", "description": "Outside city limits"}
        ]
      }
    }
  ],
  "outputs": [
    {
      "name": "predicted_price",
      "label": "Predicted Price",
      "type": "float",
      "description": "Estimated market value of the property",
      "unit": "USD",
      "format": "currency",
      "min": 50000,
      "max": 2000000
    },
    {
      "name": "confidence_interval",
      "label": "Price Range",
      "type": "object",
      "description": "95% confidence interval for the prediction",
      "schema": {
        "lower": {"type": "float", "unit": "USD", "format": "currency"},
        "upper": {"type": "float", "unit": "USD", "format": "currency"}
      }
    }
  ],
  "input_groupings": [
    {
      "grouping": "property_details",
      "description": "Physical characteristics of the property",
      "inputs": ["square_feet", "bedrooms"]
    },
    {
      "grouping": "location_info",
      "description": "Geographic and neighborhood information",
      "inputs": ["location"]
    }
  ],
  "python_environment": {
    "type": "conda",
    "name": "house-predictor",
    "requirements": [
      "scikit-learn>=1.1.0",
      "pandas>=1.4.0",
      "numpy>=1.21.0"
    ]
  }
}
```

### Example 2: Advanced NLP Model

A sophisticated text classification model with conditional inputs:

```json
{
  "model_id": "multilingual-sentiment-v3",
  "model_name": "Multilingual Sentiment Analysis",
  "model_version": "3.2.1",
  "description_short": "BERT-based sentiment analysis supporting 15 languages",
  "description": "A fine-tuned multilingual BERT model for sentiment analysis. Supports 15 languages with confidence scoring and entity-aware sentiment detection.",
  "links": [
    {
      "name": "Model Card",
      "url": "https://huggingface.co/sentiment-multilingual-v3",
      "description": "Detailed model performance and training information"
    },
    {
      "name": "Research Paper",
      "url": "https://arxiv.org/abs/2023.12345",
      "description": "Technical details and evaluation results"
    }
  ],
  "tags": {
    "domain": "natural_language_processing",
    "task": "sentiment_analysis",
    "framework": "transformers",
    "language": "multilingual",
    "difficulty": "advanced",
    "performance": "production_ready"
  },
  "inputs": [
    {
      "name": "text",
      "label": "Text to Analyze",
      "type": "string",
      "required": true,
      "description": "Input text for sentiment analysis (max 512 tokens)",
      "constraints": {
        "max_length": 2000,
        "min_length": 1
      }
    },
    {
      "name": "language",
      "label": "Language",
      "type": "category",
      "required": false,
      "description": "Text language (auto-detected if not specified)",
      "default": "auto",
      "constraints": {
        "options": [
          {"value": "auto", "label": "Auto-detect"},
          {"value": "en", "label": "English"},
          {"value": "es", "label": "Spanish"},
          {"value": "fr", "label": "French"},
          {"value": "de", "label": "German"},
          {"value": "it", "label": "Italian"}
        ]
      }
    },
    {
      "name": "confidence_threshold",
      "label": "Confidence Threshold",
      "type": "float",
      "required": false,
      "description": "Minimum confidence for classification (lower values = more sensitive)",
      "default": 0.7,
      "unit": "probability",
      "constraints": {
        "min": 0.1,
        "max": 0.99,
        "step": 0.01
      }
    },
    {
      "name": "enable_entity_sentiment",
      "label": "Entity-level Sentiment",
      "type": "boolean",
      "required": false,
      "description": "Extract sentiment for individual entities/topics",
      "default": false
    },
    {
      "name": "entity_types",
      "label": "Entity Types to Analyze",
      "type": "category",
      "required": false,
      "description": "Types of entities to extract sentiment for",
      "depends_on": {
        "input_name": "enable_entity_sentiment",
        "mapping": {
          "true": {
            "required": true,
            "default": ["PERSON", "ORG"],
            "constraints": {
              "options": [
                {"value": "PERSON", "label": "People"},
                {"value": "ORG", "label": "Organizations"},
                {"value": "PRODUCT", "label": "Products"},
                {"value": "EVENT", "label": "Events"}
              ],
              "multiple": true
            }
          },
          "false": {
            "required": false
          }
        }
      }
    }
  ],
  "outputs": [
    {
      "name": "sentiment",
      "label": "Overall Sentiment",
      "type": "string",
      "description": "Primary sentiment classification",
      "options": [
        {"value": "positive", "label": "Positive", "description": "Positive sentiment detected"},
        {"value": "negative", "label": "Negative", "description": "Negative sentiment detected"},
        {"value": "neutral", "label": "Neutral", "description": "Neutral or mixed sentiment"}
      ]
    },
    {
      "name": "confidence",
      "label": "Confidence Score",
      "type": "float",
      "description": "Model confidence in the sentiment prediction",
      "unit": "probability",
      "format": "percentage",
      "min": 0.0,
      "max": 1.0
    },
    {
      "name": "sentiment_scores",
      "label": "Detailed Scores",
      "type": "object",
      "description": "Raw scores for each sentiment class",
      "schema": {
        "positive": {"type": "float", "min": 0.0, "max": 1.0},
        "negative": {"type": "float", "min": 0.0, "max": 1.0},
        "neutral": {"type": "float", "min": 0.0, "max": 1.0}
      }
    },
    {
      "name": "detected_language",
      "label": "Detected Language",
      "type": "string",
      "description": "Auto-detected or specified language code"
    },
    {
      "name": "entity_sentiments",
      "label": "Entity Sentiments",
      "type": "array",
      "description": "Sentiment analysis for individual entities (if enabled)",
      "item_type": "object",
      "schema": {
        "entity": {"type": "string", "description": "Entity text"},
        "type": {"type": "string", "description": "Entity type"},
        "sentiment": {"type": "string", "description": "Entity sentiment"},
        "confidence": {"type": "float", "min": 0.0, "max": 1.0}
      }
    }
  ],
  "input_presets": [
    {
      "input_preset": "analysis_modes",
      "label": "Analysis Mode",
      "description": "Common configurations for different use cases",
      "affects": ["confidence_threshold", "enable_entity_sentiment"],
      "presets": [
        {
          "name": "Quick Analysis",
          "description": "Fast sentiment detection with standard settings",
          "values": {
            "confidence_threshold": 0.5,
            "enable_entity_sentiment": false
          }
        },
        {
          "name": "Detailed Analysis",
          "description": "Comprehensive analysis with entity-level sentiment",
          "values": {
            "confidence_threshold": 0.7,
            "enable_entity_sentiment": true,
            "entity_types": ["PERSON", "ORG", "PRODUCT"]
          }
        },
        {
          "name": "High Precision",
          "description": "Conservative analysis with high confidence requirements",
          "values": {
            "confidence_threshold": 0.9,
            "enable_entity_sentiment": false
          }
        }
      ]
    }
  ],
  "input_groupings": [
    {
      "grouping": "input_text",
      "description": "Text input and language settings",
      "inputs": ["text", "language"],
      "order": 1
    },
    {
      "grouping": "analysis_settings", 
      "description": "Control analysis behavior and sensitivity",
      "inputs": ["confidence_threshold"],
      "order": 2
    },
    {
      "grouping": "entity_analysis",
      "description": "Entity-level sentiment extraction",
      "inputs": ["enable_entity_sentiment", "entity_types"],
      "collapsible": true,
      "order": 3
    }
  ],
  "python_environment": {
    "type": "conda",
    "name": "multilingual-sentiment",
    "python_version": "3.9",
    "channels": ["conda-forge", "huggingface"],
    "requirements": [
      "transformers>=4.25.0",
      "torch>=1.13.0",
      "tokenizers>=0.13.0",
      "numpy>=1.21.0"
    ],
    "pip_requirements": [
      "spacy>=3.4.0",
      "langdetect>=1.0.9"
    ]
  }
}
```

## Best Practices

Follow these guidelines to create effective and maintainable model metadata.

### ✅ Do's

**Clear and Descriptive Names**
- Use descriptive `model_id` values: `"fraud-detection-v2"` not `"model2"`
- Write clear `label` text: `"Learning Rate"` not `"lr"`
- Provide helpful `description` text for all inputs and outputs

**Sensible Defaults and Constraints**
- Set reasonable default values that work for most use cases
- Use appropriate constraints to prevent invalid inputs
- Include units for numeric parameters when applicable

**Logical Organization**
- Group related parameters using `input_groupings`
- Order groups logically (basic → advanced)
- Use presets for common parameter combinations

**Comprehensive Documentation**
- Include external links to papers, documentation, and examples
- Add tags for discoverability and categorization
- Write descriptions that explain both what and why

### ❌ Don'ts

**Poor Naming and Documentation**
- Don't use technical jargon without explanation
- Avoid abbreviations in user-facing text
- Don't assume users understand model internals

**Invalid Constraints**
- Don't set `min > max` for numeric constraints
- Avoid overly restrictive regex patterns
- Don't create circular dependencies in conditional inputs

**UI Anti-patterns**
- Don't put too many inputs in one group (max 5-7)
- Avoid deeply nested conditional dependencies
- Don't hide important parameters in collapsed sections by default

### Performance Considerations

**Large Models**
- Use appropriate environment specifications for memory requirements
- Consider input validation to prevent out-of-memory errors
- Document expected execution times

**Complex UIs**
- Limit conditional depth to 2-3 levels maximum
- Use presets to simplify complex parameter spaces
- Group advanced options separately from basic settings

## Troubleshooting

Common issues and their solutions when working with model metadata.

### Validation Errors

**"Required field missing"**
```
Error: Missing required field 'description' in input 'learning_rate'
```
**Solution**: Ensure all required fields are present in each input/output definition.

**"Invalid constraint definition"**
```
Error: Constraint 'min' (1.0) cannot be greater than 'max' (0.5)
```
**Solution**: Check numeric constraints for logical consistency.

**"Circular dependency detected"**
```
Error: Circular dependency: input_a depends on input_b which depends on input_a
```
**Solution**: Review conditional input dependencies for circular references.

### Runtime Issues

**"Environment not found"**
```
Error: Conda environment 'my-model-env' not found
```
**Solution**: Verify environment configuration and ensure it exists or can be created.

**"Invalid input value"**
```
Error: Input 'learning_rate' value 1.5 exceeds maximum constraint 1.0
```
**Solution**: Check constraint definitions and default values for consistency.

### UI Problems

**"Input not displayed"**
Check:
- Input is listed in `input_groupings` if groupings are used
- Conditional dependencies are correctly configured
- Input name matches exactly (case-sensitive)

**"Preset not working"**
Verify:
- Preset `affects` array includes all relevant input names
- Input names in preset `values` match exactly
- Conditional preset dependencies are properly configured

### Performance Issues

**"Slow UI rendering"**
Common causes:
- Too many conditional inputs with complex dependencies
- Large number of category options
- Deeply nested input groupings

**Solutions**:
- Simplify conditional logic
- Use string arrays instead of rich objects for simple categories
- Reduce number of inputs per group

## Migration Guide

Guidelines for updating metadata when upgrading Chanterelle versions or model implementations.

### Version 2.x to 3.x

**Breaking Changes**
- `python_environment.requirements_file` → `python_environment.requirements`
- `input.validation` → `input.constraints`
- `output.possible_values` → `output.options`

**Migration Steps**
1. Update field names according to mapping above
2. Convert requirements file references to inline arrays
3. Test all conditional dependencies still work
4. Validate updated metadata against new schema

### Model Version Updates

**Semantic Versioning**
- **Major** (X.0.0): Breaking changes to inputs/outputs
- **Minor** (0.X.0): New features, backward-compatible
- **Patch** (0.0.X): Bug fixes, no interface changes

**Backward Compatibility**
When updating model versions:
- Maintain existing input names and types
- Add new inputs as optional with sensible defaults
- Deprecate rather than remove outputs immediately
- Document changes in model description

### Best Practices for Updates

**Version Control**
- Keep metadata files in version control
- Tag releases with corresponding model versions
- Maintain changelog for metadata changes

**Testing**
- Validate metadata after each change
- Test UI generation with updated metadata
- Verify all presets and conditional logic work
- Check environment setup on clean systems

**Documentation Updates**
- Update model description for significant changes
- Refresh external links and references
- Add migration notes for breaking changes
- Update tags to reflect new capabilities

This comprehensive reference provides everything needed to create effective model metadata for Chanterelle. Use the examples as starting points and refer to the troubleshooting section when issues arise.
