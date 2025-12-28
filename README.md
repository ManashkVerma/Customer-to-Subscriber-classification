# App-data-not-enrollment-detection

This project analyzes **user behavior data from a FinTech mobile application** to understand:

* user engagement patterns
* enrollment behavior
* feature usage
* time-to-enrollment distribution

The goal is to **prepare clean features for machine learning** and extract **insights through exploratory data analysis (EDA)**.

---

## 📂 Dataset Description

The main dataset contains **50,000 user records** with the following information:

| Column                 | Description                             |
| ---------------------- | --------------------------------------- |
| `user`                 | Unique user ID                          |
| `first_open`           | Timestamp when the app was first opened |
| `dayofweek`            | Day of the week (0–6)                   |
| `hour`                 | Hour of app usage                       |
| `age`                  | User age                                |
| `screen_list`          | List of screens visited                 |
| `numscreens`           | Number of screens visited               |
| `minigame`             | Minigame usage (binary)                 |
| `used_premium_feature` | Premium feature usage (binary)          |
| `enrolled`             | Enrollment status (target variable)     |
| `enrolled_date`        | Enrollment timestamp                    |
| `liked`                | App liked (binary)                      |

Additional file:

* **`top_screens.csv`** → list of most frequently used screens

---

## ⚙️ Libraries Used

```python
numpy
pandas
matplotlib
seaborn
python-dateutil
```

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 Histograms of numerical features

* Visualized distributions for:

  * age
  * number of screens
  * hour
  * day of week
  * feature usage

```python
plt.hist(data_num.iloc[:, i], bins=unique_values)
```

---

### 🔹 Correlation with target variable

Measured how numerical features relate to **enrollment**:

```python
data_num.corrwith(data.enrolled).plot.bar()
```

---

### 🔹 Correlation heatmap

Visualized inter-feature relationships using Seaborn:

```python
sns.heatmap(corr, mask=mask, cmap=cmap)
```

---

## ⏱️ Time-to-Enrollment Analysis

Calculated **hours between first app open and enrollment**:

```python
data['difference'] = (
    data.enrolled_date - data.first_open
).dt.total_seconds() / 3600
```

### Visualizations:

* Full distribution
* Zoomed ranges:

  * 0–1000 hours
  * 0–50 hours

---

## 🚨 Enrollment Label Correction

Users enrolling **after 48 hours** were relabeled as **not enrolled**:

```python
data.loc[data['difference'] > 48, 'enrolled'] = 0
```

This creates a more **realistic prediction target**.

---

## 📱 Screen Usage Features

Loaded top screens:

```python
topscreens = pd.read_csv('./top_screens.csv').top_screens.values
```

These screens can be used for:

* feature engineering
* one-hot encoding
* behavioral modeling

---

## ✅ Final Dataset

After cleaning and feature engineering:

* Time-based leakage removed
* Enrollment labels corrected
* Ready for **machine learning models**

---

## 🚀 Next Steps

* Encode `screen_list` using top screens
* Train ML models (Logistic Regression, Random Forest, XGBoost)
* Evaluate performance (ROC-AUC, precision/recall)
* Feature importance analysis

---

## 🧠 Key Takeaways

* Most enrollments happen within **48 hours**
* Screen usage patterns strongly influence conversion
* Time-based feature leakage must be handled carefully

---

## 📌 How to Run

```bash
pip install numpy pandas matplotlib seaborn python-dateutil
python analysis.py
```

---


