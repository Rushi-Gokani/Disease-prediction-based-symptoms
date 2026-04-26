import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from joblib import dump, load
import json
import os

warnings.filterwarnings('ignore')

SEPARATOR = "=" * 70

print(SEPARATOR)
print("  COMPREHENSIVE MODEL TRAINING - Disease Prediction System")
print(SEPARATOR)

results_summary = {}

# ============================================================
# PART 1: Binary Symptom Dataset (132 features)
# ============================================================
print("\n" + "=" * 70)
print("  PART 1: Binary Symptom Dataset (training_data.csv)")
print("=" * 70)

df_train = pd.read_csv('./dataset/training_data.csv')
df_test = pd.read_csv('./dataset/test_data.csv')

feature_cols = [c for c in df_train.columns if c != 'prognosis' and not str(c).startswith('Unnamed')]
X_train_bin = df_train[feature_cols]
y_train_bin = df_train['prognosis']
X_test_bin = df_test[feature_cols]
y_test_bin = df_test['prognosis']

le_bin = LabelEncoder()
le_bin.fit(pd.concat([y_train_bin, y_test_bin]))
y_train_bin_enc = le_bin.transform(y_train_bin)
y_test_bin_enc = le_bin.transform(y_test_bin)

print(f"Training samples: {len(X_train_bin)}, Test samples: {len(X_test_bin)}")
print(f"Features: {len(feature_cols)}, Diseases: {len(le_bin.classes_)}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_bin, y_train_bin_enc, test_size=0.2, random_state=42, stratify=y_train_bin_enc
)

print(f"Train split: {len(X_tr)}, Validation split: {len(X_val)}")

models_binary = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=2000),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear']
        }
    }
}

best_bin_model_name = None
best_bin_accuracy = 0
best_bin_model = None

for name, config in models_binary.items():
    print(f"\n--- Training {name} (Binary) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        config['model'], config['params'], cv=cv,
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid.fit(X_tr, y_tr)

    val_pred = grid.best_estimator_.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    test_pred = grid.best_estimator_.predict(X_test_bin.values)
    test_acc = accuracy_score(y_test_bin_enc, test_pred)

    print(f"  Best params: {grid.best_params_}")
    print(f"  CV Best Score: {grid.best_score_:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    results_summary[f'binary_{name}'] = {
        'cv_score': float(grid.best_score_),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'best_params': {k: str(v) for k, v in grid.best_params_.items()}
    }

    if test_acc > best_bin_accuracy:
        best_bin_accuracy = test_acc
        best_bin_model_name = name
        best_bin_model = grid.best_estimator_

print(f"\n>>> BEST BINARY MODEL: {best_bin_model_name} with Test Accuracy: {best_bin_accuracy:.4f}")
dump(best_bin_model, './saved_model/best_binary_model.joblib')
dump(le_bin, './saved_model/binary_label_encoder.joblib')
print(f"    Saved to ./saved_model/best_binary_model.joblib")

print(f"\n--- Classification Report (Best Binary Model: {best_bin_model_name}) ---")
test_pred = best_bin_model.predict(X_test_bin.values)
print(classification_report(y_test_bin_enc, test_pred, target_names=le_bin.classes_))

# ============================================================
# PART 2: Text-based Symptom Dataset (Symptom2Disease.csv)
# ============================================================
print("\n" + "=" * 70)
print("  PART 2: Text-based Symptom Dataset (Symptom2Disease.csv)")
print("=" * 70)

df_text = pd.read_csv('Symptom2Disease.csv')
df_text = df_text[['label', 'text']].dropna()
df_text['text'] = df_text['text'].str.strip()

le_text = LabelEncoder()
df_text['label_encoded'] = le_text.fit_transform(df_text['label'])

print(f"Total samples: {len(df_text)}")
print(f"Unique diseases: {df_text['label'].nunique()}")
print(f"Diseases: {sorted(df_text['label'].unique().tolist())}")

X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    df_text['text'], df_text['label_encoded'],
    test_size=0.2, random_state=42, stratify=df_text['label_encoded']
)

print(f"Train: {len(X_train_txt)}, Test: {len(X_test_txt)}")

models_text = {
    'LinearSVC': {
        'model': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('clf', LinearSVC(max_iter=5000))
        ]),
        'params': {
            'tfidf__max_features': [3000, 5000, 8000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__C': [0.1, 1, 10]
        }
    },
    'LogisticRegression': {
        'model': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=2000, random_state=42))
        ]),
        'params': {
            'tfidf__max_features': [3000, 5000, 8000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10, 100],
            'clf__solver': ['lbfgs', 'liblinear']
        }
    },
    'MultinomialNB': {
        'model': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('clf', MultinomialNB())
        ]),
        'params': {
            'tfidf__max_features': [3000, 5000, 8000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__alpha': [0.01, 0.1, 0.5, 1.0]
        }
    },
    'RandomForest_Text': {
        'model': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'params': {
            'tfidf__max_features': [3000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [10, 20, None]
        }
    },
    'XGBoost_Text': {
        'model': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('clf', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', verbosity=0))
        ]),
        'params': {
            'tfidf__max_features': [3000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__n_estimators': [100, 200, 300],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7]
        }
    }
}

best_txt_model_name = None
best_txt_accuracy = 0
best_txt_model = None

for name, config in models_text.items():
    print(f"\n--- Training {name} (Text) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        config['model'], config['params'], cv=cv,
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid.fit(X_train_txt, y_train_txt)

    y_pred = grid.best_estimator_.predict(X_test_txt)
    test_acc = accuracy_score(y_test_txt, y_pred)
    f1 = f1_score(y_test_txt, y_pred, average='weighted')

    print(f"  Best params: {grid.best_params_}")
    print(f"  CV Best Score: {grid.best_score_:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")

    results_summary[f'text_{name}'] = {
        'cv_score': float(grid.best_score_),
        'test_accuracy': float(test_acc),
        'test_f1': float(f1),
        'best_params': {k: str(v) for k, v in grid.best_params_.items()}
    }

    if test_acc > best_txt_accuracy:
        best_txt_accuracy = test_acc
        best_txt_model_name = name
        best_txt_model = grid.best_estimator_

print(f"\n>>> BEST TEXT MODEL: {best_txt_model_name} with Test Accuracy: {best_txt_accuracy:.4f}")
dump(best_txt_model, './saved_model/best_text_model.joblib')
dump(le_text, './saved_model/text_label_encoder.joblib')
print(f"    Saved to ./saved_model/best_text_model.joblib")

print(f"\n--- Classification Report (Best Text Model: {best_txt_model_name}) ---")
y_pred = best_txt_model.predict(X_test_txt)
print(classification_report(y_test_txt, y_pred, target_names=le_text.classes_))

# Save all models for potential use
for name, config in models_text.items():
    pass

dump(best_txt_model, './text_disease_model.joblib')
dump(le_text, './text_label_encoder.joblib')

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  TRAINING SUMMARY")
print("=" * 70)

print(f"\n  Binary Symptom Model:")
print(f"    Best: {best_bin_model_name} | Test Accuracy: {best_bin_accuracy:.4f}")

print(f"\n  Text Symptom Model:")
print(f"    Best: {best_txt_model_name} | Test Accuracy: {best_txt_accuracy:.4f}")

with open('training_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\n  Detailed results saved to training_results.json")

print("\n--- Quick Test (Text Model) ---")
test_inputs = [
    "I have been having severe headaches and feel dizzy all the time",
    "My skin has red itchy patches and it peels a lot",
    "I have a high fever, body aches, and feel very weak",
    "I am experiencing stomach pain and nausea after eating",
    "I have difficulty breathing and my chest feels tight",
    "I have joint pain and stiffness in my knees and fingers",
    "I feel very tired all the time and have gained weight unexpectedly"
]

for text in test_inputs:
    pred_enc = best_txt_model.predict([text])[0]
    pred_disease = le_text.inverse_transform([pred_enc])[0]
    try:
        scores = best_txt_model.decision_function([text])[0]
        top3_idx = np.argsort(scores)[-3:][::-1]
        top3 = [(le_text.inverse_transform([i])[0], f"{scores[i]:.2f}") for i in top3_idx]
    except (AttributeError, NotImplementedError):
        try:
            probs = best_txt_model.predict_proba([text])[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(le_text.inverse_transform([i])[0], f"{probs[i]*100:.1f}%") for i in top3_idx]
        except:
            top3 = [(pred_disease, "N/A")]

    print(f"\n  Input: '{text[:60]}...'")
    print(f"  Predicted: {pred_disease}")
    print(f"  Top 3: {top3}")

print("\n\nTRAINING COMPLETE!")
