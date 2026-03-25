import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("../data/dataset.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy * 100)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation (better evaluation)
scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation Accuracy:", scores.mean() * 100)

# Save model
joblib.dump(model, "../model/model.pkl")

print("\nModel saved successfully!")