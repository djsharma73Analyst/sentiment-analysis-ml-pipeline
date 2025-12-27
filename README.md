# End-to-End Sentiment Analysis ML Pipeline

This project demonstrates a **complete end-to-end Machine Learning workflow** for text sentiment analysis — from data preprocessing and model training to model serialization and deployment using Streamlit.

The goal of this project is not just prediction accuracy, but understanding **how ML models move from notebooks to real-world applications**.

---

Project Overview

The system takes text input and predicts its sentiment:
- **Positive**
- **Negative**
- **Neutral**

It supports:
- Single-text predictions
- Batch predictions via CSV upload

---

## Key Concepts Covered

- Text preprocessing and normalization
- Feature engineering using TF-IDF
- Supervised machine learning (Logistic Regression)
- Model serialization using `pickle`
- Separation of training and inference
- Batch and real-time prediction workflows
- Deployment-ready architecture (Streamlit-based)

---

##  Tech Stack

- **Python**
- **pandas**
- **scikit-learn**
- **NLTK**
- **Streamlit**

---

##  Project Structure
├── notebook.ipynb # Jupyter notebook for training and experimentation
├── app.py # Streamlit app for inference
├── model.pkl # Serialized trained model
├── requirements.txt # Python dependencies
└── README.md # Project documentation

This project is designed as a foundation for building production-ready NLP systems. The following enhancements represent the next level of maturity for this pipeline:

Model Quality Improvements

Train on larger, real-world datasets to improve generalization.

Perform systematic evaluation using train/validation splits, confusion matrices, and F1-score.

Conduct error analysis to understand common failure cases.

Advanced NLP Models

Replace TF-IDF features with Transformer-based models (e.g., DistilBERT) for improved contextual understanding.

Compare classical ML models with deep learning models in terms of accuracy, latency, and cost.

Production Readiness

Version trained model artifacts and configurations for reproducibility.

Add input validation, confidence thresholds, and logging for robust inference.

Separate training and inference workflows for scalable deployment.

Cloud & MLOps Integration

Store datasets and model artifacts in cloud storage (e.g., Azure Blob Storage).

Automate retraining pipelines based on new data availability.

Monitor prediction confidence and data drift over time.

API-First Deployment

Expose inference via a REST API (FastAPI) and use Streamlit as a frontend interface.

Support high-volume batch and real-time prediction workloads.

