# Loan Approval Prediction with Machine Learning

This project explores how machine learning can be used to support loan approval decisions using historical loan application data. The focus is on understanding the factors that influence loan approval and evaluating different models. 

## Project Overview

The analysis compares two tree-based models:
- Random Forest
- XGBoost (baseline and tuned)

Special attention is given to class imbalance, model evaluation metrics, and the trade-offs between precision and recall.

## Dataset

The dataset is sourced from Kaggle and is publicly available at: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data.

It contains roughly contains 45,000 loan applications and the target variable is `loan_status' (1 = approved, 0 = rejected).

The dataset includes a mix of financial, credit, and demographic features.

## Methodology

The project includes the following steps:
1. Data loading and initial checks  
2. Exploratory data analysis (EDA)  
3. Feature selection and encoding  
4. Train–test split with class imbalance handling  
5. Model training and evaluation  
6. Comparison of model performance  

Class imbalance is handled using SMOTE for Random Forest and `scale_pos_weight` for XGBoost

Model performance is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

## Results Summary
- Random Forest achieved higher precision and accuracy but lower recall
- XGBoost achieved higher recall and ROC AUC, making it better at identifying approved loans.
- The best model selection will depend on business priorities, with Random Forest as a better option for Risk control and XGBOOST more suited towards identifying more eligible applicants 

## Ethical and Regulatory Note

This project was completed as an academic exercise using a publicly available dataset and so some features are included for exploratory purposes only. In real-world applications, sensitive attributes are often restricted or excluded for ethical practice.

## To run this project:
1. Clone the repository  
2. Install the required dependencies  
3. Open and run the Jupyter notebook  

```bash
pip install -r requirements.txt
