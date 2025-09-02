import pandas as pd
import joblib
import shap

# 1. Load your trained model
model = joblib.load("best_baseline_model.pkl")

# 2. Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 3. Drop target column
X = data.drop(columns=["customerID", "Churn"])  # remove ID + target

# 4. Ensure TotalCharges is numeric
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
X = X.fillna(0)

# 5. Create SHAP explainer
explainer = shap.Explainer(model, X)

# 6. Save it
joblib.dump(explainer, "explainer.pkl")

print("✅ Explainer saved as explainer.pkl")
