# Customer Churn Prediction System

## 📌 Project Overview

Customer churn is a critical business problem where companies aim to identify customers who are likely to stop using their services. This project builds an **end-to-end machine learning pipeline** to predict customer churn using structured customer data, with a strong focus on **business impact, class imbalance handling, and interpretability**.

The solution uses **Logistic Regression with threshold tuning** to maximize churn recall, ensuring that most at-risk customers are identified early.

---

## 🎯 Objectives

* Predict whether a customer will churn
* Handle class imbalance effectively
* Optimize recall for churn customers
* Interpret key drivers behind churn
* Build a clean, production-ready ML pipeline

---

## 🗂️ Dataset Description

The dataset contains customer demographics, account information, and service usage details.

**Target Variable:**

* `Churn` (Yes / No)

**Key Feature Groups:**

* Customer tenure and billing information
* Contract and payment methods
* Internet and service add-ons
* Demographic indicators

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand churn behavior and guide modeling decisions.

### Key Insights:

* Churn rate is **highest in the first year** of customer tenure
* **Month-to-month contracts** have significantly higher churn
* Customers without **Online Security / Tech Support** churn more
* **Higher monthly charges** increase churn probability
* Dataset is **imbalanced (~27% churn)**, making accuracy unreliable

These insights directly informed feature selection, evaluation metrics, and threshold tuning.

---

## ⚙️ Machine Learning Pipeline

The project uses a fully reproducible Scikit-Learn pipeline:

1. **Data Preprocessing**

   * Handling missing values (`TotalCharges`)
   * Binary encoding of Yes/No variables
   * One-Hot Encoding for categorical features

2. **Modeling**

   * Logistic Regression with `class_weight="balanced"`
   * Pipeline + ColumnTransformer architecture

3. **Evaluation Metrics**

   * ROC–AUC (primary metric)
   * Recall for churn class
   * Confusion Matrix

4. **Threshold Tuning**

   * Optimized decision threshold (≈ 0.41)
   * Increased churn recall from 80% → **87%**

---

## 📊 Model Performance

### Logistic Regression (Final Model)

* **ROC–AUC:** 0.836
* **Churn Recall:** 0.87
* **Accuracy:** 0.69 (expected due to imbalance)

### Random Forest (Comparison Model)

* **ROC–AUC:** 0.814
* **XGBoost:** 0.825

**Final Choice:** Logistic Regression

* Better ROC–AUC
* Higher churn recall
* Fully interpretable
* Business-friendly deployment

---

## 🔎 Feature Importance (Interpretability)

Top features **increasing churn**:

* Month-to-month contract
* Fiber optic internet service
* Electronic check payment
* No online security / tech support
* High monthly charges

Top features **reducing churn**:

* Two-year contracts
* Automatic bank transfer payments
* Online security & tech support
* Long customer tenure

This confirms the model learned meaningful, real-world customer behavior.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── Data.csv
│   └── processed/
│       └── processed.csv
│
├── models/
│   └── churn_logreg_model.joblib
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run data preprocessing:

```bash
python src/data_preprocessing.py
```

3. Train the model:

```bash
python src/train.py
```

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Model deployment using FastAPI
* Monitoring churn drift over time
* Cost-sensitive learning

---

## 🧠 Key Takeaway

This project demonstrates how **business-aware ML decisions**—such as handling imbalance, optimizing recall, and interpreting features—are often more important than using complex models.

---

## 📬 Contact

Feel free to reach out for feedback or collaboration.

---

⭐ If you found this project useful, consider starring the repository!
