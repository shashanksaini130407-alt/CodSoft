import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/saved_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# CUSTOMER DATA
# =========================

customer = {
    "CreditScore":650,
    "Gender":1,
    "Age":40,
    "Tenure":5,
    "Balance":60000,
    "NumOfProducts":2,
    "HasCrCard":1,
    "IsActiveMember":1,
    "EstimatedSalary":80000,
    "Geography_Germany":0,
    "Geography_Spain":1
}

customer_df = pd.DataFrame([customer])

# =========================
# SCALE DATA
# =========================

customer_scaled = scaler.transform(customer_df)

# =========================
# PREDICT
# =========================

prediction = model.predict(customer_scaled)

probability = model.predict_proba(customer_scaled)

print()
print("Prediction Result")

if prediction[0] == 1:
    print("Customer will churn")
else:
    print("Customer will stay")

print()
print("Probability = ",probability)