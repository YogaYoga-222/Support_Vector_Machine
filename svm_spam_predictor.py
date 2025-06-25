import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("/home/stemland/Downloads/spam.csv", encoding='latin1') 

# Rename and clean up column names if needed
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels (Ham = 0, Spam = 1)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])  # Ham: 0, Spam: 1

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_encoded']

# Split the data
X_train, X_test, y_train, y_test, msg_train, msg_test = train_test_split(
    X, y, df['message'], test_size=0.2, random_state=42
)

# Train SVM model with linear kernel
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Summary
ham_count = sum(y_pred == 0)
spam_count = sum(y_pred == 1)
print("\n--- Prediction Summary ---")
print(f"Ham Messages Predicted: {ham_count}")
print(f"Spam Messages Predicted: {spam_count}")

# Sample Predictions - clean output
print("\n--- Sample Predictions ---\n")
sample_df = pd.DataFrame({
    'Message': msg_test.iloc[:5].values,
    'Actual': ['Ham' if label == 0 else 'Spam' for label in y_test.iloc[:5]],
    'Predicted': ['Ham' if label == 0 else 'Spam' for label in y_pred[:5]]
})
sample_df['Message'] = sample_df['Message'].str.slice(0, 60) + '...'
print(sample_df.to_markdown(index=False))

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))



