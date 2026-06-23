# ==================================
# CUSTOMER CHURN MODEL TRAINING
# ==================================

# Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv("dataset/Churn_Modelling.csv")

# ==================================
# REMOVE USELESS COLUMNS
# ==================================

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# ==================================
# ENCODE GENDER
# ==================================

gender_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# ==================================
# ENCODE GEOGRAPHY
# ==================================

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# ==================================
# INPUT AND OUTPUT
# ==================================

X = df.drop(columns=["Exited"])
y = df["Exited"]

# ==================================
# SPLIT DATA
# ==================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==================================
# FEATURE SCALING
# ==================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================
# CREATE MODELS
# ==================================

log_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# ==================================
# TRAIN MODELS
# ==================================

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# ==================================
# PREDICTIONS
# ==================================

log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# ==================================
# ACCURACY
# ==================================

log_acc = accuracy_score(y_test, log_pred)
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)

print("Logistic =", log_acc)
print("Random Forest =", rf_acc)
print("Gradient Boosting =", gb_acc)

# ==================================
# FIND BEST MODEL
# ==================================

scores = {
    "Logistic": log_acc,
    "RandomForest": rf_acc,
    "GradientBoosting": gb_acc
}

best = max(scores, key=scores.get)

print()
print("Best Model =", best)

if best == "Logistic":
    model = log_model
    prediction = log_pred
elif best == "RandomForest":
    model = rf_model
    prediction = rf_pred
else:
    model = gb_model
    prediction = gb_pred

# ==================================
# SAVE MODEL
# ==================================

joblib.dump(model, "models/saved_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ==================================
# REPORT
# ==================================

print()
print(classification_report(y_test, prediction))

# ==================================
# CONFUSION MATRIX
# ==================================

cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn")
plt.show()

print()
print("Training Completed")