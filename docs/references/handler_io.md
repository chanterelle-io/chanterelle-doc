# handler_io.py

## Overview
The `handler_io.py` file is designed to implement handler functions for machine learning models. These functions are used to load the model, process input data, make predictions, and format outputs. The file is structured to ensure modularity and reusability.

## File Structure
The file contains the following key functions:

**Required Handler Functions**: `model_fn`, `input_fn`, `predict_fn`, `output_fn`.
**Optional Handler Functions**: `init_resources_fn`.

### 1. `model_fn`
**Purpose**: Load the machine learning model from the specified directory.

**Parameters**: 
- `model_dir` - Directory containing the model files.

**Example**:
```python
def model_fn(model_dir):
    import joblib
    import os
    model_path = os.path.join(model_dir, 'model.joblib')
    predictor = joblib.load(model_path)
    return predictor
```

### 2. `init_resources_fn` (optional)
**Purpose**: Load additional resources (e.g., lookup tables) during initialization.
**Parameters**: 
- `model_dir` - Directory containing the model and additional resources.

**Implementation**:
- Load JSON files or other resources.
- Handle exceptions gracefully.

**Example**:
```python
def init_resources_fn(model_dir):
    import json
    import os
    resources = {}
    
    resources_path = os.path.join(model_dir, 'lookup_dict.json')
    if os.path.exists(resources_path):
        with open(resources_path, 'r') as f:
            resources['lookup_dict'] = json.load(f)
    
    return resources
```

### 3. `input_fn`
**Purpose**: Transform raw input data into a format suitable for the model.
**Parameters**: 
- `request_data` - Input data from the request 
- `resources` (optional) - Additional resources loaded by `init_resources_fn`.

**Implementation**:
- Normalize and validate input data.
- Extract and build features.

**Example**:
```python
def input_fn(request_data, resources):
    // ... Extract features from request_data
    return features_data
```

### 4. `predict_fn`
**Purpose**: Run predictions using the loaded model.
**Parameters**:
- `input_data` - The processed input data from `input_fn`.
- `model` - The loaded model from `model_fn`.
- `resources` (optional) - Additional resources loaded by `init_resources_fn`.

**Implementation**:
- Make predictions using the model.
- Handle sensitivity analysis.

**Example**:
```python
output = model.predict(input_data)
```

### 5. `output_fn`
**Purpose**: Format the model predictions into a response format for the frontend.
**Parameters**:
- `predictions` - The output from `predict_fn`.
- `original_data` - The original request data for reference.
- `resources` (optional) - Additional resources loaded by `init_resources_fn`.
**Output**
- the output is a (JSON-like) list of sections following the same structure as [The model findings JSON file](./references/model_findings.md).

**Implementation**:
- Output predictions and calculate derived metrics.
- Create sections for insights display.

**Example**:
```python
def output_fn(predictions, original_data, resources):
    #  Transform predictions from Celsius to Fahrenheit
    predicted_Celsius = predictions['predicted_Celsius']
    predicted_Fahrenheit = [(pred * 9.0 / 5.0) + 32 for pred in predicted_Celsius]

    # Create response structure
    response = []
    # Add section for a table displaying predictions in Celsius and Fahrenheit
    results_section = {
        "type": "section",
        "id": "results",
        "title": "Prediction Results",
        "color": "green",  # Optional: color for section header
        "description": "Model prediction for temperature in Celsius and Fahrenheit.",
        "items": [
            {
                "type": "table",
                "id": "prediction_table",
                "title": "Prediction Summary",
                "data": {
                    "columns": [
                        {"header": "Metric", "field": "metric"},
                        {"header": "Value", "field": "value"},
                        {"header": "Unit", "field": "unit"}
                    ],
                    "rows": [
                        {
                            "metric": "Temperature (Celsius)",
                            "value": f"{float(predicted_Celsius):.4f}",
                            "unit": "°C"
                        },
                        {
                            "metric": "Temperature (Fahrenheit)",
                            "value": f"{float(predicted_Fahrenheit):.4f}",
                            "unit": "°F"
                        }
                    ]
                }
            }
        ]
    }
    response.append(results_section)
    return response
```

## Best Practices
- **Error Handling**: Use `try-except` blocks to handle exceptions gracefully.
- **Validation**: Ensure input data is consistent and meets the model's requirements.

## Chanterelle Integration
The `handler_io.py` file is typically used in conjunction with a base handler (`python_handler_base.py`) that calls its functions automatically. Ensure the file is placed in the correct project directory and the model files are accessible.

