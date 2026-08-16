# Credit Card Fraud Detection - Simple College Project

This project uses Python to identify fraudulent credit card transactions.

The project uses two models:

- Logistic Regression as a simple baseline
- A small neural network using scikit-learn's `MLPClassifier`

TensorFlow, SMOTE, Streamlit, saved model files, source folders, and output folders are intentionally
not used. All analysis, charts, modeling, and conclusions are inside one executed Jupyter notebook.

## Repository Structure

```text
Credit-Card-Fraud-Detection-Simple/
|-- data/
|   |-- Download from kaggle
|-- credit_card_fraud_detection_simple.ipynb
|-- README.md
|-- requirements.txt
```

## Dataset

The dataset contains anonymized credit card transactions with the following columns:

- `Time`: seconds from the first transaction
- `V1` to `V28`: anonymized PCA features
- `Amount`: transaction amount
- `Class`: target column where `0` means normal and `1` means fraud

The full compressed dataset is included at `data/creditcard.csv.zip`. Pandas reads the ZIP directly,
so it does not need to be extracted.

## Notebook Workflow

1. Import libraries
2. Load and inspect the dataset
3. Check missing values and duplicates
4. Visualize fraud distribution
5. Compare transaction amounts by class
6. Study hourly fraud patterns
7. Create a correlation heatmap and PCA scatter plot
8. Split the data into training and testing sets
9. Balance only the training data with simple under-sampling
10. Train Logistic Regression and a small neural network
11. Compare precision, recall, F1, ROC-AUC, and PR-AUC
12. Display confusion matrices, ROC curve, and precision-recall curve
13. Explain business meaning and limitations

## How to Run on a Laptop

Open a terminal in the project folder and install the packages:

```powershell
pip install -r requirements.txt
```

Start JupyterLab:

```powershell
jupyter lab
```

Open `credit_card_fraud_detection_simple.ipynb`, then select **Run All Cells**.

## Why Accuracy Is Not Enough

Fraud is extremely rare, so a model can have very high accuracy while missing most fraud cases.
This project focuses on:

- Precision: how many flagged transactions are actually fraud
- Recall: how many fraud transactions the model finds
- F1 score: balance between precision and recall
- ROC-AUC: overall class-separation ability
- PR-AUC: performance focused on the rare fraud class

## Results from the Executed Notebook

The cleaned dataset contains `283,726` transactions and `473` fraud cases. The original imbalanced
test set is used for final evaluation.

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Small Neural Network | `0.4491` | `0.7895` | `0.5725` | `0.9641` | `0.7011` |
| Logistic Regression | `0.2058` | `0.8211` | `0.3291` | `0.9611` | `0.6576` |

The small neural network produced the better F1 score and reduced false positives compared with
Logistic Regression. Logistic Regression found slightly more fraud cases but generated more false alerts.

## Laptop-Friendly Choices

- The full dataset is used for EDA and testing.
- Only the training data is under-sampled before model training.
- The neural network contains two small hidden layers.
- Early stopping prevents unnecessary training.
- No model or chart files are saved outside the notebook.

## GitHub Note

The compressed dataset is large. Use GitHub Desktop or Git from the terminal when publishing this
repository instead of uploading the dataset through the GitHub website.

## Skills Demonstrated

- Data cleaning
- Exploratory data analysis
- Data visualization
- Imbalanced classification
- Logistic Regression
- Neural networks
- Model evaluation
- Business interpretation

## Possible Improvements

- Tune the classification threshold based on fraud investigation cost
- Compare random forest or gradient boosting models
- Add cross-validation
- Test SMOTE as an alternative balancing method
- Add SHAP explanations
