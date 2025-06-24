# 📈 Customer Churn Prediction with Advanced ML Models

## A Comprehensive Machine Learning Pipeline for Telco Customer Churn

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Reports and Results](#-reports-and-results)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Running with Docker](#-running-with-docker)

---

## 🎯 Introduction

This project presents a robust and modular machine learning pipeline designed to predict customer churn in a telecommunications dataset. Customer churn is a critical business problem, as retaining existing customers is often more cost-effective than acquiring new ones. By accurately identifying customers at risk of churning, businesses can implement proactive retention strategies, leading to improved customer lifetime value and reduced revenue loss.

This repository encapsulates best practices in MLOps, focusing on clear project structure, automated processes, and comprehensive reporting.

## ⚠️ Problem Statement

Customer churn is a significant challenge for telecommunication companies. Identifying customers who are likely to discontinue their service is vital for implementing targeted retention campaigns. This project aims to build a predictive model that can accurately classify customers into 'churn' or 'no churn' categories, providing actionable insights to business stakeholders.

## ✨ Features

* **Automated Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical features.
* **Multiple Advanced Model Training:** Supports training and hyperparameter tuning for a diverse set of powerful classification algorithms:
    * Logistic Regression
    * Random Forest
    * XGBoost
    * LightGBM
    * CatBoost
* **Automated Hyperparameter Tuning:** Utilizes `GridSearchCV` to find optimal parameters for each model.
* **Best Model Selection:** Automatically identifies and selects the best-performing model based on ROC AUC score.
* **Model Persistence:** Saves the best-performing model for future use.
* **Comprehensive Reporting:** Generates a detailed Markdown report with key metrics (Accuracy, ROC AUC, Precision, Recall, F1-Score) and visualizations (ROC Curve, Confusion Matrix) for the best model.
* **Reproducibility:** Uses a fixed random state and manages dependencies via `requirements.txt`.

## 📂 Project Structure
```text
Churn-prediction/
├── data/
│   └── telco.csv           # Raw customer churn dataset
├── models/
│   └── catboost_model.joblib # Saved best model (e.g., CatBoost)
├── notebooks/
│   └── (optional: EDA, experimentation notebooks)
├── reports/
│   ├── images/             # Generated plots (ROC curve, confusion matrix)
│   │   ├── roc_curve_catboost.png
│   │   └── confusion_matrix_catboost.png
│   └── model_performance_summary_YYYY-MM-DD_HHMMSS.md # Detailed performance report
├── src/
│   ├── config.py           # Global configurations and hyperparameters
│   ├── data_preprocessing.py # Functions for data loading, cleaning, encoding
│   ├── adv_models.py       # Advanced model definitions and tuning logic
│   ├── evaluate.py         # Model evaluation and report generation functions
│   ├── model.py            # Script for loading and evaluating a saved model
│   └── model_selection.py  # Orchestrates model training, selection, and reporting
├── main.py                 # Main entry point for training and evaluation
└── requirements.txt        # Python package dependencies
```


## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9 or higher
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MGhassanCs/Churn-Prediction
    cd Churn-prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your dataset:**
    Ensure you have the `telco.csv` dataset inside the `data/` directory.
    
### Running the Project

To run the project, use the CLI commands below:

- **Train and select the best model:**
  ```bash
  python main.py train --data data/telco.csv
  ```
- **Evaluate a saved model:**
  ```bash
  python main.py evaluate --data data/telco.csv --model models/catboost_model.joblib
  ```
  If you omit the `--model` argument, the most recently saved model in the `models/` directory will be used automatically.

**Note:** Running `python main.py` without a command will show the help message. Always specify `train` or `evaluate`.

**What do these commands do?**
- `train`: Trains all models, selects the best, saves it, and generates a report.
- `evaluate`: Loads a saved model and evaluates it on the test set (no retraining).

## 📊 Reports and Results

Upon successful execution of the training command, a new folder named `reports/` will be created in your project's root directory.

Inside `reports/`, you will find:
* A markdown file (`model_performance_summary_YYYY-MM-DD_HHMMSS.md`) containing a comprehensive summary of the best model's performance, including key metrics and interpretations.
* An `images/` subdirectory, which contains the `roc_curve_*.png` and `confusion_matrix_*.png` plots specific to the best-performing model.

These reports are designed to be human-readable and provide a clear overview of the model's effectiveness in predicting churn.

## 🛠️ Technologies Used

* **Python** (3.9+)
* **pandas** - Data manipulation and analysis
* **numpy** - Numerical computing
* **scikit-learn** - Machine learning library (Logistic Regression, RandomForest, GridSearchCV, preprocessing, metrics)
* **xgboost** - Gradient Boosting library (XGBClassifier)
* **lightgbm** - Gradient Boosting library (LGBMClassifier)
* **catboost** - Gradient Boosting library (CatBoostClassifier)
* **matplotlib** - Plotting library
* **seaborn** - Statistical data visualization
* **joblib** - For saving and loading Python objects (models)

## 💡 Future Enhancements

* **Feature Engineering:** Explore more advanced feature creation techniques (e.g., interaction terms, polynomial features).
* **Model Interpretability:** Implement techniques like SHAP or LIME to explain model predictions for individual customers.
* **Deployment:** Containerize the model using Docker and deploy it using a framework like Flask/FastAPI or cloud services.
* **MLflow Integration:** Use MLflow for experiment tracking, model registry, and reproducible runs.
* **Automated Data Validation:** Add checks for data quality and schema validation.
* **Continuous Integration/Deployment (CI/CD):** Set up pipelines for automated testing, building, and deployment.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add Your Feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License.


## 🧪 Running Tests

To run the test suite:

```bash
pytest
```

This will automatically discover and run all tests in the `tests/` directory.

## 🐳 Running with Docker

You can build and run this project in a containerized environment using Docker.

### Build the Docker image
```bash
docker build -t churn-prediction .
```

### Run training (mount your data directory)
```bash
docker run --rm -v $(pwd)/data:/app/data churn-prediction train --data data/telco.csv
```

### Run evaluation (mount your data directory and models directory)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models churn-prediction evaluate --data data/telco.csv --model models/catboost_model.joblib
```

- The `-v` flag mounts your local `data` and `models` directories into the container so the code can access your dataset and save/load models.
- You can use any of the CLI commands described above inside the container.

## 🛠️ Troubleshooting

- **Training is slow:** Training all models with GridSearchCV can take 10–30+ minutes. For faster testing, reduce the number of models or hyperparameters in `src/config.py`.
- **File not found:** Make sure `data/telco.csv` exists and is mounted correctly (especially in Docker).
- **Docker resource limits:** If Docker is slow, increase CPU/RAM allocation in Docker Desktop settings.
- **Permission errors:** On some systems, you may need to adjust file permissions or run Docker with `sudo`.
- **Windows users:** Replace `$(pwd)` with `%cd%` in Docker volume mount commands.