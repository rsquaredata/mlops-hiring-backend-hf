---
title: MLOps Hiring Backend
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# MLOps Hiring Backend API

Production-ready FastAPI backend serving hiring probability predictions.

## Tech Stack

- FastAPI
- Scikit-learn
- MongoDB Atlas (Cloud)
- Docker
- HuggingFace Spaces
- GitHub Actions (CI/CD)

## Live Endpoint

POST /predict

Backend deployed on HuggingFace Spaces.

## Features

- Trained ML model (class imbalance handled)
- Probability-based predictions
- Cloud database logging
- Secret management via environment variables
- Fully containerized
- Automated deployment via GitHub Actions

## Architecture

See [meta repository](https://github.com/rsquaredata/mlops-hiring) for full system architecture.
