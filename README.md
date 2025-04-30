<img src="https://github.com/mlop-ai/mlop/raw/refs/heads/main/design/favicon.svg?sanitize=true" alt="logo" height="80">

[![stars](https://img.shields.io/github/stars/mlop-ai/mlop)](https://github.com/mlop-ai/mlop/stargazers)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb)
[![pypi](https://img.shields.io/pypi/v/mlop)](https://pypi.org/project/mlop/)
[![license](https://img.shields.io/github/license/mlop-ai/mlop)](https://github.com/mlop-ai/mlop/blob/main/LICENSE)
<!-- [![build](https://img.shields.io/github/actions/workflow/status/mlop-ai/mlop/mlop.yml)](https://github.com/mlop-ai/mlop/actions/workflows/mlop.yml) -->

**mlop** is a Machine Learning Operations (MLOps) framework. It provides [self-hostable superior experimental tracking capabilities and lifecycle management for training ML models](https://github.com/mlop-ai/server). To get started, [try out our introductory notebook](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb) or [get an account with us today](https://app.mlop.ai/auth/sign-up)!

## 🎥 Demo

**mlop** adopts a KISS philosophy that allows it to outperform all other tools in this category. Supporting high and stable data throughput should be *THE* top priority for efficient MLOps.
<video loop src='https://github.com/user-attachments/assets/efd9720e-6128-4278-85ec-ee6139a851af' alt="demo" width="1200" style="display: block; margin: auto;"></video>

<p align="center">
<strong>mlop</strong> logger (bottom left) v. a conventional logger (bottom right)
</p>

## 🚀 Getting Started

Start logging your experiments with **mlop** in 4 simple steps:

1. Get an account at [app.mlop.ai](https://app.mlop.ai/auth/sign-up)
2. Install our Python SDK. Within a Python environment, open a Terminal window and paste in the following,
```bash
pip install "mlop[full]"
```
or, for the latest nightly,
```bash
pip install "mlop[full] @ git+https://github.com/mlop-ai/mlop.git"
```
3. Log in to your [mlop.ai](https://app.mlop.ai/o) account from within the Python client,
```python
import mlop
mlop.login()
```
4. Start logging your experiments by integrating **mlop** to the scripts, as an example,
```python
import mlop

config = {'lr': 0.001, 'epochs': 1000}
run = mlop.init(project="title", config=config)

# insert custom model training code
for i in range(config['epochs']):
    run.log({"val/loss": 0})

run.finish()
```
And... profit! The script will redirect you to the webpage where you can view and interact with the run. The dashboard allows you to easily compare time series data and can provide actionable insights for your training.

These steps are described in further detail in [introductory tutorial](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb) and [torch tutorial](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/torch.ipynb).  
You may also learn more about **mlop** by checking out our [documentation](https://docs.mlop.ai/).

## 🫡 Vision

**mlop** is a platform built for and by ML engineers, supported by [our community](https://discord.gg/ybfVZgyFCX)! We were tired of the current state of the art in ML observability tools, and this tool was born to help mitigate the inefficiencies - specifically, we hope to better inform you about your model performance and training runs; and actually **save you**, instead of charging you, for your precious compute time! 

🌟 Be sure to star our repos if they help you ~
