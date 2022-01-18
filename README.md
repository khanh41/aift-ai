# AI Fitness Trainer - AIFT
![docker build](https://github.com/khanh41/aift-ai/actions/workflows/dev-deploy-docker.yml/badge.svg)

## Installation ⚡️
### Requires
- Python: 3.7~3.8

Install with poetry:
~~~
pip install poetry
poetry install
~~~

## Deployment app ⛄️
## Deployment with Docker 🐳
Docker build and run with Dockerfile:
~~~
sudo docker build -t aift-ai .
sudo docker run -it -d aift-ai
~~~

## Tree directory 🌗 
~~~
.
├── data_loader          - load data or model.
├── preprocessing        - preprocessing data.
├── figures              - draw (ignore).
├── metrics              - metrics for model, etc.
├── base_model           - model machine learning setup
├── trainers             - model machine learning training.
├── utils                - tools format, lint, test, etc.
├── resources            - image, audio, csv, etc. (ignore)
├── pyproject.toml       - dependencies and package.
└── server.py            - application creation and configuration.
~~~
