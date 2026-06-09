# student-performance-prediction-system
ML-powered system to predict student graduation and segment performance using Gradient Boosting &amp; K-Means clustering. Built with Streamlit.

# 🎓 Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent Machine Learning system that predicts student graduation status and segments academic performance using **Gradient Boosting** and **K-Means Clustering**. Built with **Streamlit** for an interactive dashboard experience.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Graduation Prediction** | Predict if a student will **GRADUATE** or **NOT GRADUATE** (G3 ≥ 10) using Gradient Boosting |
| 📊 **Performance Segmentation** | Group students into **4 performance categories** using K-Means Clustering |
| 📈 **Interactive Visualizations** | Data distribution, correlation heatmaps, and cluster visualizations |
| 📋 **Dataset Overview** | Complete dataset information and statistics |

## 🤖 Machine Learning Models

### 1. Graduation Prediction (Classification)
- **Algorithm:** Gradient Boosting Classifier
- **Target:** GRADUATE (G3 ≥ 10) or NOT GRADUATE (G3 < 10)
- **Accuracy:** ~85-88%
- **AUC Score:** ~92%
- **Features:** 10 features (grades, attendance, study time, etc.)

### 2. Performance Segmentation (Clustering)
- **Algorithm:** K-Means Clustering
- **Clusters:** 4 segments
- **Metrics:**
  - Silhouette Score: 0.42
  - Davies-Bouldin Score: 0.85
- **Segments:**
  - 🌟 **EXCELLENT** - Top performers
  - ✅ **GOOD** - Good potential
  - 📘 **AVERAGE** - Needs motivation
  - ⚠️ **POOR** - Requires intensive guidance

## 📁 Project Structure
student-performance-predictor/
├── app.py # Main Streamlit application
├── classification/ # Graduation prediction model
│ ├── model_gradient_boosting.pkl
│ ├── fitur_kelulusan.pkl
│ └── mapping_kelulusan.pkl
├── clustering/ # Performance segmentation model
│ ├── model_clustering.pkl
│ ├── scaler_clustering.pkl
│ ├── fitur_clustering.pkl
│ └── nama_segment.pkl
├── train_model_classification.py # Classification model training
├── train_model_clustering.py # Clustering model training
└── requirements.txt # Dependencies

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
