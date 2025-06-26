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
- `svm_purchase_predictor.py`: Predicts if a person will purchase a product based on age, salary, and ad viewed
- `purchase_data.csv`: Dataset with personal details and purchase label

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
---

## What I Learned

* I understood how Linear SVM works for different types of data like text, eligibility, and product purchase.
* I learned to:
  * Handle both categorical and numerical data.
  * Use label encoding to convert Yes/No values into numbers.
  * Apply StandardScaler to normalize the input features.
* I practiced giving reasons for prediction, making the output more meaningful.
* I also learned how to avoid common warnings in pandas and improve my code structure.
* Overall, this task helped me understand how to apply SVM in real-world decision-making problems.

---
