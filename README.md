# Machine Learning-Based Customer Churn Prediction for Retail Banking

A binary classification pipeline that predicts whether a bank customer will churn (exit the bank), built on the Kaggle "Binary Classification with a Bank Churn Dataset" (Playground Series S4E1). The project covers EDA, categorical encoding, class-imbalance handling with SMOTE, and a comparative benchmark of six classification algorithms, evaluated on Accuracy, Precision, Recall, and F1-score.

## Resume Description (3 bullet points)

- **Built and benchmarked an end-to-end churn prediction pipeline** on a 165,034-row banking dataset (12 features: credit score, geography, age, tenure, balance, product count, etc.), performing EDA, one-hot encoding of categorical features (Geography, Gender), and correlation analysis to identify churn drivers.
- **Resolved severe class imbalance (78.8% retained vs. 21.2% churned) using SMOTE oversampling**, balancing the training set to 130,113 samples per class before a stratified 80/20 train-test split, preventing model bias toward the majority class.
- **Trained and compared 6 classifiers** (Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, LightGBM); the best model, **LightGBM, achieved 90.85% accuracy, 92.98% precision, 88.39% recall, and a 90.63% F1-score**, outperforming the baseline Logistic Regression model (69.51% accuracy) by over 21 percentage points.

## Model Comparison (Actual Evaluation Results)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.6951 | 0.6926 | 0.7022 | 0.6974 |
| K-Nearest Neighbors | 0.7016 | 0.6665 | 0.8078 | 0.7304 |
| Decision Tree | 0.8618 | 0.8570 | 0.8687 | 0.8628 |
| Random Forest | 0.8636 | 0.8676 | 0.8585 | 0.8630 |
| XGBoost (Gradient Boosting) | 0.9031 | 0.9263 | 0.8761 | 0.9005 |
| **LightGBM (Best Model)** | **0.9085** | **0.9298** | **0.8839** | **0.9063** |

## Project Overview

### 1. Dataset
- Source: Kaggle Playground Series S4E1 (synthetic, derived from the classic Bank Customer Churn dataset)
- Train set: 165,034 rows × 14 columns | Test set: 110,023 rows × 13 columns
- Target variable: `Exited` (1 = churned, 0 = retained)
- Class distribution: 130,113 retained (78.84%) vs. 34,921 churned (21.16%) — an imbalanced classification problem
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

### 2. Data Preprocessing
- Dropped non-predictive identifier columns: `CustomerId`, `Surname`
- Verified zero missing values across all 14 columns
- One-hot encoded `Geography` (France/Germany/Spain) and `Gender` (Male/Female) using `pd.get_dummies(drop_first=True)` to avoid the dummy variable trap
- Visualized feature distributions (histograms), class counts for Gender/Geography, and a correlation heatmap

### 3. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to the training features/target to synthetically balance the minority (churned) class
- Result: balanced dataset of 130,113 samples per class (260,226 total)
- Performed an 80/20 train-test split (`random_state=42`) on the SMOTE-resampled data

### 4. Model Training & Evaluation
Six classifiers were trained and evaluated on identical train/test splits using **Accuracy, Precision, Recall, and F1-score**:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree Classifier
4. Random Forest Classifier (`n_estimators=100`, `max_depth=5`)
5. XGBoost Classifier (`binary:logistic`)
6. LightGBM Classifier — **best performer**

### 5. Final Model & Inference
- LightGBM was selected as the production model based on its top F1-score (0.9063) and highest accuracy (90.85%)
- Generated churn predictions on the held-out Kaggle test set (110,023 customers) and exported to `submission.csv`

## Tech Stack
- **Language**: Python
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Modeling**: Scikit-learn (Logistic Regression, KNN, Decision Tree, Random Forest), XGBoost, LightGBM
- **Evaluation**: Scikit-learn metrics (accuracy_score, precision_score, recall_score, f1_score)

## Key Takeaways
- Tree-based ensemble methods (LightGBM, XGBoost) substantially outperformed linear/distance-based models (Logistic Regression, KNN) on this tabular dataset, improving F1-score by ~21 points over the baseline.
- SMOTE oversampling was essential given the ~4:1 class imbalance, ensuring the model didn't simply default to predicting the majority "retained" class.
- LightGBM's gradient-boosted decision trees captured non-linear feature interactions (e.g., between Age, NumOfProducts, and IsActiveMember) that simpler models missed, making it the most reliable model for identifying at-risk customers.
