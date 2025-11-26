# Run this once to train and save the model
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import shap
import numpy as np

print("Loading Telco Customer Churn dataset...")
df = pd.read_csv("churn.csv")  # Local file you just downloaded

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Clean & preprocess (handle TotalCharges as numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Features (exact columns from dataset)
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

X = df[cat_cols + num_cols]
y = df['Churn']

# Encode categorical
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print(f"Encoded features: {len(X_encoded.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, prob):.3f}")

# Save model + columns
joblib.dump(model, "churn_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")

# SHAP explainer (for explanations in app)
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "explainer.pkl")

print("✅ Model + SHAP explainer saved!")
print("📊 Ready for deployment!")