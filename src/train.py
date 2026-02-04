import pandas as pd
import joblib
import config as cg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

df = pd.read_csv(cg.PROCESSED_DATA_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# -------- Logistic Regression (Final Model) --------
logreg_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
          max_iter=1000,
          class_weight="balanced",
          random_state=42
        )),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

logreg_model.fit(X_train, y_train)



# Evaluate (default threshold)
y_pred = logreg_model.predict(X_test)
y_proba = logreg_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("=== Logistic Regression (threshold=0.5) ===")
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", cm)
print(classification_report(y_test, y_pred))

threshold = 0.41
print(f"\nLogistic Regression (threshold={threshold})")

# Threshold tuning
threshold = 0.41
y_pred_custom = (y_proba >= threshold).astype(int)

print(f"\n=== Logistic Regression (threshold={threshold}) ===")
print(classification_report(y_test, y_pred_custom))

# -------- Random Forest --------
from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]
)

rf_model.fit(X_train, y_train)

rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print("\n=== Random Forest (threshold=0.5) ===")
print("Random Forest ROC-AUC:", rf_auc)

# -------- XGBoost --------
from xgboost import XGBClassifier

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=cg.RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ]
)

xgb_model.fit(X_train, y_train)

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)

print("\n=== XGBoost (threshold=0.5) ===")
print("XGBoost ROC-AUC:", xgb_auc)


# -------- Interpretability (LogReg coefficients) --------
feature_names = logreg_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = logreg_model.named_steps["classifier"].coef_[0]

feature_importance = (
    pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    .sort_values(by="coefficient", ascending=False)
)

print("\nTop features increasing churn:")
print(feature_importance.head(10))

print("\nTop features reducing churn:")
print(feature_importance.tail(10))

# ✅ Save final chosen model (you chose LogReg)
joblib.dump(logreg_model, cg.MODEL_PATH)
print("Model saved successfully.")