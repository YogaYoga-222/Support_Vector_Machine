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

### 3. `svm_purchase_predictor.py`

- This program predicts whether a person will purchase a product or not.
- It uses data like age, estimated salary, and whether they viewed the ad.
- It uses Linear SVM to build the model.
- The program also shows why someone is likely or unlikely to purchase (age, salary, ad viewed).
- You can also give new input to check the prediction for a new person.

### 4. `svm_credictcard_fraud_detection.py`

- This program predicts whether a credit card transaction is **fraud** or **legit**.
- It uses a cleaned CSV file (`credictcard.csv`) with features like `V1`, `V2`, ..., `V28`, and `Amount`.
- The target column is `Class`, where 0 = legit and 1 = fraud.
- The model uses **SVM with RBF (Gaussian) kernel** to handle non-linear data.
- It also shows a confusion matrix and classification report to check how well the model works.

---

## Requirements

Install the required Python libraries using:

```bash
pip install pandas numpy scikit-learn tabulate matplotlib
```
---

## Files

- `svm_spam_predictor.py`: Predicts if a text message is spam or ham using Linear SVM and TF-IDF
- `spam.csv`:	Dataset with text messages labeled as 'ham' or 'spam'
- `svm_eligibility_predictor.py`:	Predicts if a person is eligible based on age, education, and experience
- `eligibility_data.csv`:	Dataset containing person details and eligibility label
- `svm_purchase_predictor.py`: Predicts if a person will purchase a product based on age, salary, and ad viewed
- `purchase_data.csv`: Dataset with personal details and purchase label
- `svm_credictcard_fraud_detection.py`: Predicts if a credit card transaction is fraud or legit using SVM (Gaussian kernel)
- `credictcard.csv`: Dataset with credit card transaction features and fraud labels

---

## Run the Script 
```bash
python3 svm_spam_predictor.py
```
```bash
python3 svm_eligibility_predictor.py
```
```bash
python3 svm_purchase_predictor.py
```
```bash
python3 svm_credictcard_fraud_detection.py
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
### svm_purchase_predictor.py
```
Model Accuracy: 100.00%

Classification Report:

               precision    recall  f1-score   support

Not Purchased       1.00      1.00      1.00         1
    Purchased       1.00      1.00      1.00         7

     accuracy                           1.00         8
    macro avg       1.00      1.00      1.00         8
 weighted avg       1.00      1.00      1.00         8


New Person Predictions:
Person 1: Won't Purchase
Person 2: Will Purchase
Person 3: Won't Purchase
```
### svm_credictcard_fraud_detection.py
```
Model Accuracy: 99.85%

Classification Report:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      9970
           1       0.73      0.80      0.76        30

    accuracy                           1.00     10000
   macro avg       0.86      0.90      0.88     10000
weighted avg       1.00      1.00      1.00     10000
```
### Confusion Matrix (Visual)

![Confusion Matrix for Fraud Detection](./Documents/GitHub/Support_Vector_Machine/fraud_confusion_matrix.png)

---

## What I Learned

* I understood how SVM works for different types of problems like spam detection, eligibility check, product purchase prediction, and fraud detection.
* I learned how to handle both text and numeric data, and how to use TF-IDF or scaling when needed.
* I used label encoding, train-test splitting, and confusion matrix to understand model performance.
* I also learned how to improve output and make the code easier to understand.
* This project helped me apply SVM to real-world tasks in a clear and simple way.
