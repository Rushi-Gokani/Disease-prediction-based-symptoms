import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  BRFSS Diabetes Prediction Model Training")
print("=" * 60)

df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"\nClass Distribution:\n{df['Diabetes_binary'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

df = df.drop_duplicates()
print(f"\nAfter removing duplicates: {df.shape}")

print(f"\nStatistical Summary:\n{df.describe()}")

X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

FEATURE_NAMES = list(X.columns)
print(f"\nFeatures ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nBefore SMOTE - Training set: {X_train.shape}")
print(f"Class distribution:\n{y_train.value_counts()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE - Training set: {X_train_resampled.shape}")
print(f"Class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

best_model_name = None
best_accuracy = 0
best_model = None
results = {}

print("\n" + "=" * 60)
print("  Training Models")
print("=" * 60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  Accuracy: {acc * 100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

print("\n" + "=" * 60)
print("  Results Summary")
print("=" * 60)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    marker = " <-- BEST" if name == best_model_name else ""
    print(f"  {name}: {acc * 100:.2f}%{marker}")

print(f"\nBest Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy")

y_pred_best = best_model.predict(X_test_scaled)
print(f"\nClassification Report ({best_model_name}):\n")
print(classification_report(y_test, y_pred_best, target_names=['No Diabetes', 'Diabetes']))

print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")

pickle.dump(best_model, open('diabetes_brfss_model.pkl', 'wb'))
pickle.dump(scaler, open('diabetes_brfss_scaler.pkl', 'wb'))
pickle.dump(FEATURE_NAMES, open('diabetes_brfss_features.pkl', 'wb'))

print("\n" + "=" * 60)
print("  Model files saved:")
print("    - diabetes_brfss_model.pkl")
print("    - diabetes_brfss_scaler.pkl")
print("    - diabetes_brfss_features.pkl")
print("=" * 60)
