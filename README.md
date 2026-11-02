# Self-Hosted LLM on Kaggle

This branch contains the Kaggle infrastructure for the self-healing CI/CD pipeline.

It packages a Qwen3.8-27B endpoint running on two NVIDIA Tesla T4 GPUs, model evaluation kernels, deployment utilities, downloaded notebook artifacts, and archived experiment logs.

## Contents

- [`kaggle/`](kaggle/) — complete Kaggle endpoint, evaluation kernels, tools, notebooks, logs, and documentation.
- [`kaggle/README.md`](kaggle/README.md) — setup, deployment, evaluation, troubleshooting, and operational guidance.

## Public notebook

[`lacrimatct/qwen3-8-27b-live-endpoint`](https://www.kaggle.com/code/lacrimact/qwen3-8-27b-live-endpoint)

## Branch purpose

`harsh/self-hosted-llm-on-kaggle` is intentionally focused on the self-hosted Kaggle LLM implementation. The main project history remains on the `main` branch.

## Requirements

- Kaggle API access and internet-enabled notebook sessions
- NVIDIA Tesla T4 ×2
- Attached `linsxav/llama-server-builder` kernel source
- Hugging Face access to download the configured GGUF model

See [`kaggle/README.md`](kaggle/README.md) for the complete workflow and safety rules.
