---
sidebar_position: 1
---

# Introduction

## What is Chanterelle?

**Chanterelle** is a lightweight desktop application designed for **data scientists** and **ML engineers** to easily **test, present and share models and findings** in a structured and interactive way.
It allows model developers to:
- 📄 **Describe models** using structured metadata
- 📊 **Show insights and findings** through an optional report page
- 🧠 **Interact with the model** via a user-defined input/output interface

## Why use Chanterelle?

- **Local, simple, and secure** – No need to deploy to the cloud, especially before production.
- **Share models easily** – Present models, results, and visual findings with minimal setup
- **Structure your work** – Organize metadata, reports, and interfaces in a clear, reusable format
- **Collaborate like code** – Project files are JSON + Python, so teams can version them in Git, push to shared repos, and review changes just like with source code
- **Integrate with your workflow** – Fully compatible with existing Python environments and model pipelines

## How It Works

Chanterelle is built around three core components, all configurable by the model developer:
### 1. **Model Metadata**

Specify inputs, outputs, and UI settings using a JSON configuration.

➡️ [See metadata structure](./references/model_meta.md)
### 2. **Model Functions**

Write your core Python functions to:
- Load and prepare your model
- Transform inputs
- Run predictions 
- Return outputs and, optionally, generate dynamic visualizations or plots

These functions allow Chanterelle to serve as a lightweight local UI for testing and showcasing your model. Preparing these functions also makes it easier to deploy models elsewhere (e.g. AWS SageMaker).

➡️ [See model functions](#)
### 3. **Insights & Findings (Optional)**

You can include a dedicated page for:
- Model results, KPIs, and benchmark comparisons
- Visual findings like feature importance plots, confusion matrices, partial dependence plots or cohort analyses.

All content is defined via a JSON structure linked to static or dynamic visual outputs.  

➡️ [See findings structure](#)