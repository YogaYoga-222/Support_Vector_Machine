import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
df = pd.read_csv("/home/stemland/Downloads/eligibility_data.csv") 

# Features and target
X = df[['Age', 'Education_Level', 'Experience']]
y = df['Eligibility']  # 'Eligible' or 'Not Eligible'

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train SVM model (linear kernel)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Show accuracy and classification report
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f} %\n")
print("Classification Report:\n")
print(classification_report(y_test_labels, y_pred_labels))

# Function for predicting new samples
def predict_eligibility(age, edu_level, experience):
    sample_df = pd.DataFrame([[age, edu_level, experience]], columns=['Age', 'Education_Level', 'Experience'])
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)
    return le.inverse_transform(prediction)[0]

# Sample predictions
print("Sample 1 Prediction:", predict_eligibility(22, 2, 0))
print("Sample 2 Prediction:", predict_eligibility(30, 4, 5))
print("Sample 3 Prediction:", predict_eligibility(18, 1, 0))
