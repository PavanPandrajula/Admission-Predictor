import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------
# Parameters
n_rows = 10000
Country = ["USA", "INT"]

# --------------------
# Generate features
Data = pd.DataFrame({
    "Application_number": np.random.randint(100000, 200000, size=n_rows),
    "GRE": np.random.randint(260, 340, size=n_rows),
    "SOP": np.round(np.random.uniform(1, 5, size=n_rows), 1),
    "PS": np.round(np.random.uniform(1, 5, size=n_rows), 1),
    "LOR": np.round(np.random.uniform(1, 5, size=n_rows), 1),
    "Research": np.random.choice([0,1], size=n_rows),
    "IELTS": np.random.choice([0,1], size=n_rows),
    "Country": np.random.choice(Country, size=n_rows),
    "Experience": np.random.randint(1,3, size=n_rows)
})


# --------------------
# One-Hot Encode Country
encoder = OneHotEncoder(sparse_output=False)
country_encoded = encoder.fit_transform(Data[['Country']])
country_columns = encoder.get_feature_names_out(['Country'])
Data[country_columns] = country_encoded
Data = Data.drop(columns=['Country'])

# --------------------
# Scale numeric features
scaler = MinMaxScaler()
Data[['GRE','PS','SOP','LOR']] = scaler.fit_transform(Data[['GRE','PS','SOP','LOR']])

#Target Variable
Data["Admit"] = ((Data["GRE"] + Data["SOP"] + Data["PS"] + Data["Research"] + Data["LOR"] + Data["IELTS"] + Data["Experience"]) > 4.5).astype(int)

print(Data.head(200))
# --------------------
# Features and target
X = Data[['GRE','SOP','PS','LOR','Research','IELTS','Country_USA','Country_INT']]
y = Data['Admit']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# --------------------
# Define models (ensure probability=True for SVM)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

# --------------------
# Train each model and store probabilities
probabilities = []

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    probabilities.append(probs)
    print(f"{name} trained successfully.")

# --------------------
# Average probabilities (soft voting)
avg_probs = np.mean(probabilities, axis=0)

# Final prediction: class 1 if avg probability >= 0.5 else 0
final_pred = (avg_probs >= 0.5).astype(int)

# --------------------
# Evaluation
print("=== Soft Voting Ensemble ===")
print("Accuracy:", accuracy_score(y_test, final_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_pred))
print("Classification Report:\n", classification_report(y_test, final_pred))

# --------------------
joblib.dump(models, "ensemble_models.pkl")

