import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

print("Loading Symptom2Disease.csv...")
df = pd.read_csv("Symptom2Disease.csv")
df = df[["label", "text"]].dropna()

print(f"Total samples: {len(df)}")
print(f"Unique diseases: {df['label'].nunique()}")
print(f"Diseases: {sorted(df['label'].unique().tolist())}")

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_encoded"], test_size=0.2, random_state=42, stratify=df["label_encoded"]
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )),
    ("clf", LinearSVC(C=1.0, max_iter=2000))
])

print("\nTraining TF-IDF + LinearSVC model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

dump(pipeline, "text_disease_model.joblib")
dump(label_encoder, "text_label_encoder.joblib")

print("\nModel saved to: text_disease_model.joblib")
print("Label encoder saved to: text_label_encoder.joblib")

print("\n--- Testing with sample inputs ---")
test_inputs = [
    "I have been having severe headaches and feel dizzy all the time",
    "My skin has red itchy patches and it peels a lot",
    "I have a high fever, body aches, and feel very weak",
    "I am experiencing stomach pain and nausea after eating",
    "I have difficulty breathing and my chest feels tight"
]

for text in test_inputs:
    pred_encoded = pipeline.predict([text])[0]
    pred_disease = label_encoder.inverse_transform([pred_encoded])[0]
    decision_scores = pipeline.decision_function([text])[0]
    top_3_indices = np.argsort(decision_scores)[-3:][::-1]
    top_3 = [(label_encoder.inverse_transform([i])[0], f"{decision_scores[i]:.2f}") for i in top_3_indices]
    print(f"\n  Input: '{text[:60]}...'")
    print(f"  Predicted: {pred_disease}")
    print(f"  Top 3: {top_3}")

print("\nTraining complete!")
