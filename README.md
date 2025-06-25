## SVM Spam Detection (`svm_spam_predictor.py`)

### Dataset
- **File Name**: `spam.csv`
- **Description**: This file contains text messages labeled as either `ham` (not spam) or `spam`.

---

### Goal
Use **Support Vector Machine (SVM)** to build a model that predicts whether a given message is spam or not.

---

### Requirements

Make sure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn tabulate
```
---

### What the Code Does

* Loads the dataset spam.csv
* Preprocesses the data (keeps only required columns and cleans missing values)
* Converts the text messages into numerical format using TF-IDF
* Splits the data into training and testing sets
* Trains an SVM model (Linear SVM)
* Predicts spam or ham messages
* Shows:
  * Model accuracy
  * Total predicted spam and ham messages
  * Few sample predictions
  * Full classification report
 
---

### Sample Output
```
Model Accuracy: 97.94%

--- Prediction Summary ---
Ham Messages Predicted: 984
Spam Messages Predicted: 131

--- Sample Predictions ---
| Message                                                         | Actual   | Predicted   |
|:----------------------------------------------------------------|:---------|:------------|
| Funny fact Nobody teaches volcanoes 2 erupt, tsunamis 2 aris... | Ham      | Ham         |
| I sent my scores to sophas and i had to do secondary applica... | Ham      | Ham         |
| We know someone who you know that fancies you. Call 09058097... | Spam     | Spam        |
| Only if you promise your getting out as SOON as you can. And... | Ham      | Ham         |
| Congratulations ur awarded either å£500 of CD gift vouchers ... | Spam     | Spam        |

--- Classification Report ---
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       965
        spam       0.98      0.86      0.92       150

    accuracy                           0.98      1115
   macro avg       0.98      0.93      0.95      1115
weighted avg       0.98      0.98      0.98      1115
```
---

## What I Learned

* I learned what SVM is and how it is used in text classification.
* I understood how to convert messages into numbers using TF-IDF.
* I trained an SVM model to predict spam messages.
* At first, the output was messy. Then I worked on improving the output and made it cleaner and easier to understand.
