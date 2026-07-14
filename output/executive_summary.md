# Credit Card Fraud Detection Executive Summary

## Data Quality
- Raw transactions: 284,807
- Clean transactions: 283,726
- Duplicates removed: 1,081
- Fraud rate: 0.1667%

## Model Recommendation
- Recommended model: Logistic Regression + SMOTE
- Estimated cost: $12,750.00
- Precision: 0.8750
- Recall: 0.7368

## Business Takeaways
- Missing fraud cases is much more expensive than manually reviewing a few extra flagged transactions.
- Threshold tuning matters more than plain accuracy in highly imbalanced fraud detection.
- A production workflow should monitor precision drift and retrain when cardholder behavior shifts.

## Visualization Notes
- figures/fraud_distribution.png
- figures/amount_by_class.png
- figures/hourly_pattern.png
- figures/correlation_heatmap.png

## Model Comparison
```text
                      model  threshold  accuracy  precision  recall  f1_score  roc_auc  pr_auc  true_negative  false_positive  false_negative  true_positive  estimated_cost
Logistic Regression + SMOTE     1.0000    0.9994     0.8750  0.7368    0.8000   0.9654  0.7042          56641              10              25             70         12750.0
              Random Forest     0.6285    0.9994     0.9437  0.7053    0.8072   0.9464  0.8020          56647               4              28             67         14100.0
        Deep Neural Network     1.0000    0.9992     0.8049  0.6947    0.7458   0.9641  0.6915          56635              16              29             66         14900.0
```
