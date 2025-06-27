# SCT_DS_3
A decision tree classifier to predict whether a customer will purchase a product or service based on their demographical and behavioral data.
# 🧐 Bank Marketing Decision Tree Classifier

This project uses a **Decision Tree Classifier** to predict whether a customer will subscribe to a **term deposit product** (i.e., purchase a financial service) based on their demographic and behavioral data. The analysis is performed using the popular **Bank Marketing dataset** from the UCI Machine Learning Repository.

---

## 📂 Dataset Overview

* **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
* **Filename**: `bank-full.csv`
* **Records**: 45,211
* **Features**: 16 (demographic, financial, and contact-related)
* **Target**: `y` (Binary: `yes` or `no` — whether the client subscribed to a term deposit)

---

## 📊 Features Used

* `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`
* `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
* `y` — the target label (1 = subscribed, 0 = not subscribed)

---

## 🔧 Project Pipeline

### Step 1: Upload or Mount Dataset

* Option A: Upload `bank-full.csv` manually to Colab
* Option B: Mount Google Drive and load from path

### Step 2: Data Preprocessing

* Load data with `pandas`
* Encode all categorical variables using `LabelEncoder`

### Step 3: Split Dataset

* Split into train and test sets (80/20 split)

### Step 4: Model Training

* Train a `DecisionTreeClassifier` using `sklearn`

### Step 5: Evaluation

* Generate accuracy score
* View classification report
* Display confusion matrix

### Step 6: Prediction on New Customer

* Define customer attributes
* Encode values using trained encoders
* Predict using trained classifier
* Output: `yes` or `no`

### Step 7: Visualization (Optional)

* Use `sklearn.tree.plot_tree` to visualize decision rules

---

## 🔍 Sample Prediction

Predict if a customer with the following details will subscribe:

```python
new_customer = pd.DataFrame([{
    'age': 35,
    'job': label_encoders['job'].transform(['technician'])[0],
    'marital': label_encoders['marital'].transform(['married'])[0],
    'education': label_encoders['education'].transform(['secondary'])[0],
    'default': label_encoders['default'].transform(['no'])[0],
    'balance': 1000,
    'housing': label_encoders['housing'].transform(['yes'])[0],
    'loan': label_encoders['loan'].transform(['no'])[0],
    'contact': label_encoders['contact'].transform(['cellular'])[0],
    'day': 15,
    'month': label_encoders['month'].transform(['jul'])[0],
    'duration': 120,
    'campaign': 2,
    'pdays': 999,
    'previous': 0,
    'poutcome': label_encoders['poutcome'].transform(['unknown'])[0],
}])

prediction = clf.predict(new_customer)
prediction_label = label_encoders['y'].inverse_transform(prediction)
print("🔮 Prediction for new customer:", prediction_label[0])
```

**Output:**

```
Prediction for new customer: yes
```

---

## 📊 Evaluation Metrics

* **Accuracy Score**: \~X% (depending on split and features)
* **Classification Report**: Includes precision, recall, F1-score
* **Confusion Matrix**: Visualized with heatmap

---

## 📈 Visualization

* The trained decision tree is plotted with:

```python
plot_tree(clf, feature_names=X.columns, class_names=label_encoders['y'].classes_, filled=True)
```

This allows you to understand the key splits and decision paths.

---

## 🌐 Technologies Used

* Python 3.x
* Google Colab (runtime)
* Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

---

## 👀 Project Structure

```
.
├── bank-full.csv
├── decision_tree_classifier.ipynb
├── README.md
```

---

## 🤝 Acknowledgements

* UCI Machine Learning Repository for the dataset
* scikit-learn for ML tools
* Google Colab for hosted execution environment

---


