import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

# =======================
# Step 1: Load the Dataset
# =======================
df = pd.read_csv("test.csv")

# Drop irrelevant columns
df.drop(columns=['id', 'Name'], inplace=True)

# Rename target column
df.rename(columns={'Have you ever had suicidal thoughts ?': 'SuicidalThoughts'}, inplace=True)

# Encode target values
df['SuicidalThoughts'] = df['SuicidalThoughts'].str.strip().map({'Yes': 1, 'No': 0})

# Drop rows with missing target
df.dropna(subset=['SuicidalThoughts'], inplace=True)

# Fill missing values with mode
df = df.fillna(df.mode().iloc[0])

# ===============================
# 📊 PLOT 1: Target Distribution
# ===============================
plt.figure(figsize=(5, 4))
sns.countplot(x='SuicidalThoughts', data=df)
plt.title("Distribution of Suicidal Thoughts")
plt.xticks([0, 1], ['No', 'Yes'])
plt.ylabel("Count")
plt.xlabel("Suicidal Thoughts")
plt.tight_layout()
plt.show()

# ===============================
# Step 2: Encode Categorical Columns
# ===============================
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'SuicidalThoughts']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ===============================
# Step 3: Train/Test Split
# ===============================
X = df.drop('SuicidalThoughts', axis=1)
y = df['SuicidalThoughts']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Step 4: Train Logistic Regression
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# Step 5: Evaluation
# ===============================
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 📊 ROC Curve & AUC
# ===============================
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
