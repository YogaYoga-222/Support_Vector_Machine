import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("/home/stemland/Downloads/purchase_data.csv")  

# Clean and encode the categorical data
df['Ad_Viewed'] = df['Ad_Viewed'].map({'Yes': 1, 'No': 0})
df['Purchased'] = df['Purchased'].map({'Yes': 1, 'No': 0})

# Drop rows with missing or incorrectly mapped values
df = df.dropna()

# Features and target
X = df[['Age', 'Estimated_Salary', 'Ad_Viewed']]
y = df['Purchased'].astype(int) 

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear SVM model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Print evaluation metrics
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Not Purchased", "Purchased"]))

# Make predictions on new samples
new_samples = pd.DataFrame([
    [25, 30000, 0],
    [45, 60000, 1],
    [22, 25000, 0]
], columns=['Age', 'Estimated_Salary', 'Ad_Viewed'])

new_samples_scaled = scaler.transform(new_samples)
new_preds = model.predict(new_samples_scaled)

print("\nNew Person Predictions:")
for i, pred in enumerate(new_preds):
    result = "Will Purchase" if pred == 1 else "Won't Purchase"
    print(f"Person {i+1}: {result}")
