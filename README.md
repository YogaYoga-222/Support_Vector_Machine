# Support Vector Machine (SVM) Projects

## Program Details

### 1. `svm_spam_predictor.py`

- This program predicts whether a text message is **spam** or **not spam** (ham).
- It uses **TF-IDF** to convert text into numbers and **SVM (Linear)** to classify the messages.
- It shows the accuracy, prediction summary, sample predictions, and a detailed classification report.

### 2. `svm_eligibility_predictor.py`

- This program predicts whether a person is **eligible** or **not eligible** (for a job, scholarship, etc.).
- It uses features like **age**, **education level**, and **experience** from a dataset.
- It uses **Linear SVM** to make predictions.
- You can also give your own input (age, education, experience) and check the prediction in real time.

---

## Requirements

Install the required Python libraries using:

```bash
pip install pandas numpy scikit-learn tabulate
```
---

## Files

- `svm_spam_predictor.py`: Predicts if a text message is spam or ham using Linear SVM and TF-IDF
- `spam.csv`:	Dataset with text messages labeled as 'ham' or 'spam'
- `svm_eligibility_predictor.py`:	Predicts if a person is eligible based on age, education, and experience
- `eligibility_data.csv`:	Dataset containing person details and eligibility label

---

## Run the Script 
```bash
python3 svm_spam_predictor.py
```
```bash
python3 svm_eligibility_predictor.py
```
---

## Sample Output

### svm_spam_predictor.py
```
Model Accuracy: 97.94%

--- Prediction Summary ---
Ham Messages Predicted: 984
Spam Messages Predicted: 131

--- Sample Predictions ---

| Message                                                         | Actual   | Predicted   |
|:----------------------------------------------------------------|:---------|:------------|
| Funny fact Nobody teaches volcanoes to erupt...                 | Ham      | Ham         |
| We know someone who fancies you...                              | Spam     | Spam        |
| Congratulations! You’ve won a prize...                          | Spam     | Spam        |

--- Classification Report ---
              precision    recall  f1-score   support
         ham       0.98      1.00      0.99       965
        spam       0.98      0.86      0.92       150
    accuracy                           0.98      1115
```
### svm_eligibility_predictor.py
```
Model Accuracy: 90.0 %

Classification Report:
              precision    recall  f1-score   support
    Eligible       1.00      0.75      0.86         8
Not Eligible       0.86      1.00      0.92        12
    accuracy                           0.90        20

Sample 1 Prediction: Not Eligible
Sample 2 Prediction: Eligible
Sample 3 Prediction: Not Eligible
```
---

## What I Learned

* I learned what SVM (Support Vector Machine) is and how to use it for classification problems.
* I practiced Linear SVM using real datasets for:
  * Spam detection from text messages.
  * Eligibility prediction based on personal details.
* I learned to:
  * Clean and prepare data using pandas.
  * Convert text to numbers using TF-IDF.
  * Scale numerical features.
  * Train and evaluate a model using SVM from scikit-learn.
* At first, the outputs were hard to read. I fixed and improved the formatting for better understanding.
* I also added user input support to test predictions for new data easily.
  
---
