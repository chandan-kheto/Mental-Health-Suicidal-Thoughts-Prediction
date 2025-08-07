# 🧠 Mental Health: Suicidal Thoughts Prediction

This project aims to predict whether a person has had suicidal thoughts based on lifestyle, mental health, and professional factors using a logistic regression model. The dataset is sourced from a real-world mental health survey.

---

## 📂 Dataset Info

- Source: [Kaggle - Exploring Mental Health Data](https://www.kaggle.com/datasets/adilshamim8/exploring-mental-health-data)
- Target variable: **Have you ever had suicidal thoughts?**
- Type: **Binary Classification** (Yes = 1, No = 0)
- Contains demographic + mental wellness + lifestyle features

---

## ✅ Problem Statement

Build a model that can **predict suicidal thoughts** based on features like:
- Age, Gender, Profession, Financial Stress, Family History
- Sleep Duration, Study/Work Hours, Academic Pressure, etc.

---

## 🛠️ ML Techniques Used

| Step | Description |
|------|-------------|
| Data Cleaning | Handled missing values, removed irrelevant columns |
| Preprocessing | One-Hot Encoding for categorical columns |
| Model Used | Logistic Regression (sklearn) |
| Evaluation | Accuracy, Classification Report, ROC Curve, AUC Score |

---

## 📈 Performance Metrics

- ✅ **Accuracy:** 75%+
- 📊 **Classification Report:** Precision, Recall, F1-Score for both classes
- 🔍 **ROC AUC Score:** Measures model's class-separating ability
- 📉 **ROC Curve:** Visual representation of performance on imbalanced data

---

## 📊 ROC Curve & AUC

ROC Curve was plotted to visualize how well the model distinguishes between:
- **Positive Class (1)**: Suicidal thoughts
- **Negative Class (0)**: No suicidal thoughts

> AUC closer to 1 indicates a strong model.

---

## 📦 Python Libraries Used

- `pandas`, `numpy` for data manipulation  
- `seaborn`, `matplotlib` for visualization  
- `sklearn` for preprocessing, modeling, and evaluation  

---

## 💡 Key Insights

- Sleep duration, work/study pressure, and financial stress were **important indicators**.
- Logistic regression gave reasonable performance with good AUC.
- ROC curve was helpful in evaluating the model on **imbalanced data**.

---

## 🚀 Future Improvements

- Try more powerful models: **Random Forest, XGBoost**
- Use **SHAP values** for better feature explanation
- Deploy using **Streamlit or Flask*
