# Credit Card Fraud Detection Using Deep Learning

An end-to-end fraud analytics project built on the Kaggle `creditcard.csv` dataset.  
The project cleans highly imbalanced transaction data, performs fraud-focused EDA, engineers additional behavioral features, compares baseline machine learning models, trains a deep neural network, and exposes predictions through a Streamlit app.

## Project Highlights

- Cleans transaction data and removes duplicate records before modeling
- Performs fraud-focused EDA on class balance, amount behavior, hourly activity, and feature correlation
- Engineers time-based and amount-based features on top of the PCA variables
- Handles class imbalance with `SMOTE`, class weighting, and threshold tuning
- Benchmarks `Logistic Regression`, `Random Forest`, and a `TensorFlow/Keras` neural network
- Produces business-friendly outputs, reusable artifacts, and a deployable Streamlit dashboard

## Business Problem

Banks and payment teams need to detect fraudulent transactions quickly without overwhelming analysts with too many false alarms.  
This project focuses on a practical tradeoff:

- catch as many fraud cases as possible
- minimize costly missed fraud
- keep manual review volume manageable
- compare models using operations-aware metrics instead of plain accuracy

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- TensorFlow / Keras
- Streamlit
- matplotlib
- seaborn

## Repository Structure

```text
credit_card_fraud_detection_deep_learning/
|-- data/
|   |-- README.md
|-- models/
|-- notebooks/
|-- output/
|-- src/
|   |-- fraud_pipeline.py
|   |-- streamlit_app.py
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset:

- `284,807` total transactions
- `492` fraud cases
- extremely imbalanced binary target (`Class`)

The pipeline supports either:

- `data/creditcard.csv`
- `data/creditcard.csv.zip`

You can also point to an external dataset path with `--input`.

## Project Flow

1. Data loading
2. Data cleaning
3. Exploratory data analysis
4. Feature engineering
5. Imbalance handling with `SMOTE`
6. Baseline model training
7. Deep neural network training
8. Threshold tuning and evaluation
9. Business cost analysis
10. Streamlit deployment

## How to Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full training pipeline:

```powershell
python src/fraud_pipeline.py --input "path\to\creditcard.csv.zip"
```

Example using your local dataset path:

```powershell
python src/fraud_pipeline.py --input "D:\Project resume\2.Credit Card Fraud Detection\creditcard.csv.zip"
```

Open the notebook version:

```powershell
jupyter notebook notebooks/credit_card_fraud_detection_workflow.ipynb
```

Run the app after training:

```powershell
streamlit run src/streamlit_app.py
```

## Outputs

The pipeline writes:

- `output/data_quality_summary.csv`
- `output/feature_summary.csv`
- `output/model_comparison.csv`
- `output/business_summary.csv`
- `output/executive_summary.md`
- `output/analysis_report.html`
- `output/figures/` charts when plotting libraries are installed
- `notebooks/credit_card_fraud_detection_workflow.ipynb` for a presentation-friendly walkthrough
- `models/baseline_models.pkl`
- `models/deep_learning_preprocessor.pkl`
- `models/fraud_neural_network.keras`

This repository now includes generated example outputs from a completed run so recruiters and reviewers can open the analysis report directly on GitHub or locally without retraining first.

## Evaluation Metrics

This project compares models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Estimated business cost from false positives and false negatives

## Resume-Ready Skills Demonstrated

- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Imbalanced Classification
- Supervised Learning
- Deep Learning
- Model Evaluation
- Business Impact Analysis
- Streamlit Deployment

## Suggested GitHub Repository Name

- `Credit-Card-Fraud-Detection-Deep-Learning`
- `credit-card-fraud-detection-keras`
- `fraud-detection-binary-classification`
