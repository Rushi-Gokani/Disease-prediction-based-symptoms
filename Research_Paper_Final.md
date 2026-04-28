<p align="center">
<img src="https://img.shields.io/badge/IEEE-Research%20Paper-blue?style=for-the-badge" alt="IEEE Paper"/>
<img src="https://img.shields.io/badge/ML-Healthcare-green?style=for-the-badge" alt="ML Healthcare"/>
</p>

---

<h2 align="center">Multi-Modal Disease Prediction System Using Machine Learning and Natural Language Processing: A Comprehensive Approach to Symptom-Based Diagnosis</h2>

---

<p align="center"><b>Rushi Gokani</b><sup>1</sup></p>
<p align="center"><sup>1</sup> Department of Computer Science and Engineering, [University Name], [City], India</p>
<p align="center"><i>Email: [your.email@university.edu]</i></p>

---

**Received:** March 2026 | **Revised:** April 2026 | **Accepted:** [Date] | **Published:** [Date]

---

## Abstract

Early and accurate disease prediction is essential for improving patient outcomes and reducing the burden on healthcare systems, particularly in resource-constrained settings. This paper presents a comprehensive multi-modal disease prediction system that integrates four distinct machine learning modules into a unified web-based platform: (1) a binary symptom-based classifier using Support Vector Machines (SVM) with RBF kernel achieving 100% test accuracy across 41 diseases using 132 symptom features, (2) a natural language text-based classifier using TF-IDF vectorization with Multinomial Naive Bayes (MNB) achieving 97.92% accuracy across 24 diseases, (3) a basic diabetes risk assessment using the PIMA Indians dataset (768 records, 8 features), and (4) an advanced diabetes prediction model trained on the CDC BRFSS 2015 dataset (229,474 records, 21 health indicators) with SMOTE-based class balancing, achieving 84.05% accuracy. A PyTorch-based feedforward neural network chatbot provides conversational health guidance with a 0.75 confidence threshold. Extensive hyperparameter tuning via GridSearchCV with 5-fold stratified cross-validation was employed across all modules. Comparative evaluation of six classifiers (SVM, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) demonstrates that algorithm selection is task-dependent, with SVM excelling on high-dimensional binary data and MNB outperforming complex models on small-scale text classification. Additionally, disease information is enriched using the Gemma 3 large language model via Ollama for generating structured medical content. The system is deployed as a Flask-based web application featuring automatic specialist mapping, online appointment booking with Razorpay payment integration, and role-based access control.

**Index Terms:** Disease Prediction, Machine Learning, Natural Language Processing, Support Vector Machine, Naive Bayes, TF-IDF, Symptom Analysis, Diabetes Prediction, SMOTE, Neural Network, Healthcare Informatics

---

## I. Introduction

### A. Background and Motivation

The healthcare industry faces persistent challenges in early disease detection and timely diagnosis. According to the World Health Organization, delayed diagnosis remains a significant contributor to preventable morbidity and mortality worldwide [1]. Patients frequently delay seeking medical attention due to uncertainty about the severity of their symptoms, limited access to specialists, or financial constraints. In developing countries, the physician-to-patient ratio remains critically low, creating an urgent need for intelligent decision-support systems that can assist both patients and healthcare providers in preliminary disease screening.

Machine learning (ML) offers a transformative approach to building such systems. Disease prediction from symptoms is fundamentally a multi-class classification problem where input features represent the presence or absence of specific symptoms, and the output is a predicted disease label. The challenge lies in handling the high dimensionality of symptom spaces, managing the overlap of symptoms across diseases, and accommodating the varying granularity of patient-reported symptom descriptions.

Natural Language Processing (NLP) further enhances accessibility by allowing patients to describe their symptoms in plain text, removing the barrier of medical terminology knowledge. Deep Learning techniques enable the creation of intelligent chatbots that can interact with patients and provide conversational health guidance.

### B. Problem Statement

Existing disease prediction systems typically focus on a single prediction modality -- either structured binary symptom inputs or free-text descriptions -- but rarely both. Furthermore, specialized disease prediction (such as diabetes risk assessment) is often treated as a separate system entirely. This fragmentation limits usability and fails to leverage the complementary strengths of different prediction approaches. There is a clear need for an integrated, multi-modal system that combines multiple prediction pipelines, specialist recommendations, and patient management into a unified platform.

### C. Research Objectives

The objectives of this research are:

1. To develop and evaluate multiple ML classifiers for disease prediction using binary symptom vectors across 41 disease classes.
2. To implement a natural language processing (NLP) pipeline for disease prediction from free-text symptom descriptions across 24 diseases.
3. To build specialized diabetes prediction models using both the PIMA Indians dataset (8 clinical features) and the large-scale CDC BRFSS 2015 dataset (21 health indicators, 229,474 records).
4. To design a chatbot using PyTorch neural networks for conversational health guidance.
5. To comparatively evaluate model performance across different algorithms and identify optimal models for each prediction task.
6. To integrate all models into a unified, accessible web-based application with specialist mapping and appointment management.

### D. Scope and Contributions

This paper makes the following contributions:

- **Multi-modal prediction framework:** Integration of four distinct prediction modules (binary symptom, text-based, basic diabetes, advanced diabetes) into a single platform -- a capability not found in existing systems.
- **Comprehensive benchmarking:** Comparative evaluation of 6 classifiers across 4 datasets with full hyperparameter optimization via GridSearchCV.
- **Dual-scale diabetes assessment:** Combination of clinical features (PIMA, 768 records) with population-level behavioral indicators (BRFSS, 229,474 records), demonstrating that socioeconomic and lifestyle factors significantly improve prediction.
- **LLM-enriched knowledge base:** Integration of Gemma 3 (via Ollama) for automated generation of structured disease information.
- **End-to-end clinical workflow:** Prediction to specialist mapping to appointment booking to payment processing in a single system.

### E. Paper Organization

The remainder of this paper is organized as follows: Section II reviews related work. Section III describes the datasets used. Section IV presents the methodology including preprocessing, feature engineering, and model training. Section V details the system architecture. Section VI presents results and evaluation. Section VII provides discussion and analysis. Section VIII covers the LLM integration for disease information. Section IX describes implementation details. Section X concludes with key findings and future directions.

---

## II. Literature Review

### A. Machine Learning in Healthcare

Machine learning has been extensively applied in healthcare for medical image analysis, electronic health record (EHR) mining, drug discovery, and clinical decision support [1]. Supervised learning approaches, particularly classification algorithms, have shown strong performance in disease prediction tasks where labeled training data is available.

### B. Symptom-Based Disease Prediction

Deepthi et al. [2] compared Naive Bayes, Decision Tree, and Random Forest for symptom-based disease prediction, demonstrating that Decision Tree and Random Forest achieved over 95% accuracy on structured binary symptom datasets. Kunjir et al. [10] further demonstrated the effectiveness of Decision Tree classifiers on binary symptom vectors. Zannat et al. [15] developed a comprehensive symptoms-disease dataset with 758 unique symptom-disease relationships spanning 85 diseases, achieving 98% accuracy with ensemble voting.

### C. NLP for Medical Text Classification

TF-IDF vectorization combined with traditional classifiers has been applied extensively to medical text classification. Wang and Manning [11] demonstrated that simpler models like Multinomial Naive Bayes with well-tuned TF-IDF features achieve competitive performance on moderately sized datasets with lower computational cost. Hamdi et al. [14] evaluated fine-tuned transformer models (BERT-based) with ensemble learning for disease classification from symptom text, achieving 80.56% accuracy using majority voting -- notably lower than traditional TF-IDF approaches on well-curated datasets.

### D. Diabetes Prediction

The PIMA Indians Diabetes Dataset, introduced by Smith et al. [3], has served as a standard benchmark. Sisodia and Sisodia [12] reported Logistic Regression at 78.3%, SVM at 77.6%, and Naive Bayes at 76.3% on this dataset. The CDC BRFSS dataset provides a larger-scale, population-level alternative with richer feature sets including lifestyle and behavioral indicators. Singh et al. [13] demonstrated that Random Forest and Gradient Boosting achieved 80-85% accuracy on BRFSS data with 21+ features.

### E. Class Imbalance in Medical Data

Chawla et al. [4] introduced SMOTE (Synthetic Minority Over-sampling Technique), which has become the standard approach for addressing class imbalance in medical datasets. Fernandez et al. [16] provided a comprehensive survey of SMOTE variants and their effectiveness in healthcare applications.

### F. Literature Gaps

Table I summarizes the key gaps in existing literature and how our system addresses them.

**Table I: Literature Gaps and Contributions**

| Gap in Existing Literature | Our Contribution |
|---|---|
| Single prediction modality | 4 integrated prediction modes |
| Limited diabetes features (<=8) | 21 behavioral/health indicators (BRFSS) |
| No end-to-end clinical workflow | Prediction -> Specialist -> Appointment -> Payment |
| Limited NLP for symptoms | TF-IDF pipeline achieving 97.92% on free text |
| No LLM integration | Gemma 3 for structured disease information |
| Small-scale evaluation | Up to 229,474 records with SMOTE balancing |

---

## III. Datasets

### A. Overview

Four distinct datasets were used, each serving a specific prediction module. Table II provides a summary.

**Table II: Dataset Summary**

| Dataset | Samples | Features | Classes | Type | Source |
|---|---|---|---|---|---|
| Binary Symptom | 4,920 | 132 | 41 | Binary | Public repository |
| Symptom2Disease | 1,200 | Text | 24 | NLP | Public repository |
| PIMA Diabetes | 768 | 8 | 2 | Numeric | NIDDK |
| BRFSS 2015 | 253,680 | 21 | 2 | Mixed | CDC |

### B. Binary Symptom Dataset

The binary symptom dataset consists of 4,920 training samples and 42 test samples, with 132 binary features representing the presence (1) or absence (0) of specific symptoms. The target variable (`prognosis`) contains 41 unique disease classes with approximately 120 training samples per class. Symptoms are organized across multiple clinical categories:

- **Dermatological** (18 features): itching, skin_rash, nodal_skin_eruptions, pus_filled_pimples, etc.
- **Gastrointestinal** (15 features): stomach_pain, acidity, vomiting, nausea, diarrhoea, etc.
- **Respiratory** (12 features): continuous_sneezing, breathlessness, cough, phlegm, etc.
- **Musculoskeletal** (14 features): joint_pain, muscle_wasting, muscle_weakness, stiff_neck, etc.
- **Neurological** (12 features): headache, dizziness, loss_of_balance, visual_disturbances, etc.
- **Systemic/General** (20+ features): fatigue, weight_loss, high_fever, chills, sweating, etc.
- **Hepatic/Urinary** (15+ features): dark_urine, yellowing_of_eyes, burning_micturition, etc.

The 41 disease classes span infections (Malaria, Dengue, Typhoid, Tuberculosis, etc.), liver diseases (Hepatitis A-E, Jaundice, Chronic Cholestasis), metabolic/endocrine disorders (Diabetes, Hypothyroidism, Hypoglycemia), cardiovascular conditions (Hypertension, Heart Attack), neurological disorders (Migraine, Paralysis, Vertigo), and dermatological conditions (Psoriasis, Acne, Fungal Infection).

### C. Text-Based Symptom Dataset (Symptom2Disease)

The Symptom2Disease dataset contains 1,200 free-text symptom descriptions equally distributed across 24 disease classes (50 samples per class). Each record consists of a disease label and a natural language description of symptoms. For example:

> *"I have been experiencing itchy, dry, scaly patches on my skin for the past few weeks."* -> **Psoriasis**

The perfectly balanced class distribution eliminates the need for resampling techniques on this dataset.

### D. PIMA Indians Diabetes Dataset

The PIMA dataset [3] contains 768 records from female Pima Indian patients aged 21+, with 8 clinical features: Pregnancies, Glucose (mg/dL), BloodPressure (mm Hg), SkinThickness (mm), Insulin (mu U/ml), BMI (kg/m^2), DiabetesPedigreeFunction, and Age. The class distribution is imbalanced: 500 non-diabetic (65.1%) vs. 268 diabetic (34.9%). A known data quality issue is the presence of biologically implausible zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns, representing encoded missing data.

### E. CDC BRFSS 2015 Diabetes Dataset

The Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset [8] contains 253,680 survey responses with 21 health and behavioral indicators. Features span clinical indicators (HighBP, HighChol, CholCheck, BMI, Stroke, HeartDiseaseorAttack), lifestyle factors (Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump), healthcare access (AnyHealthcare, NoDocbcCost), health status (GenHlth, MentHlth, PhysHlth, DiffWalk), and demographics (Sex, Age, Education, Income).

After deduplication (removing 24,206 duplicate rows), 229,474 unique records remain with significant class imbalance: No Diabetes 218,334 (84.6%) vs. Diabetes 35,346 (15.4%).

---

## IV. Methodology

### A. Overall Pipeline

The methodology follows a systematic pipeline applicable across all modules:

```
Raw Data -> Preprocessing -> Feature Engineering -> Train/Test Split
    -> Model Training (GridSearchCV + k-Fold CV) -> Evaluation -> Serialization
```

### B. Data Preprocessing

#### 1) Binary Symptom Data

Features are extracted by filtering out the `prognosis` target column and any unnamed columns. A sanity check ensures exactly 132 features per sample. The data is split into training (67%) and validation (33%) sets with `random_state=101`. No scaling is required as features are already binary-encoded.

#### 2) Text Symptom Data

Raw text descriptions are loaded and stripped of whitespace. Disease labels are encoded using `LabelEncoder` (0-23). The data is split 80/20 with stratification. TF-IDF vectorization is applied within a scikit-learn `Pipeline` to prevent data leakage. The TF-IDF formula used is:

```
TF-IDF(t, d) = TF(t, d) * IDF(t)

where:
    TF(t, d) = (1 + log(freq(t, d)))    [with sublinear_tf=True]
    IDF(t)   = log((1 + N) / (1 + DF(t))) + 1
```

Parameters tuned include `max_features` (3000-8000), `ngram_range` ((1,1), (1,2), (1,3)), and `stop_words='english'`.

#### 3) PIMA Diabetes Data

Biologically implausible zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are replaced with column-wise mean values. `StandardScaler` is applied for feature normalization (zero mean, unit variance). The data is split 80/20 with `random_state=7`.

#### 4) BRFSS Diabetes Data

After removing 24,206 duplicate rows (253,680 -> 229,474), a stratified 80/20 train/test split is performed. SMOTE [4] is applied exclusively to the training set to prevent data leakage, generating synthetic minority samples:

```
x_new = x_i + lambda * (x_j - x_i),    lambda ~ Uniform(0, 1)
```

where x_i is a minority sample and x_j is one of its k=5 nearest minority neighbors. This expands the training set from ~155K (imbalanced) to ~311K (balanced 50/50). `StandardScaler` is fit on training data and applied to both sets.

### C. Model Training Strategy

#### 1) Hyperparameter Optimization

All models are trained using `GridSearchCV` with `StratifiedKFold(n_splits=5, shuffle=True)` for cross-validation. This ensures each fold preserves the original class distribution. The parameter grids searched are detailed in Table III.

**Table III: Hyperparameter Search Spaces -- Binary Symptom Models**

| Model | Parameters | Search Space | Best Found |
|---|---|---|---|
| SVM | C, kernel, gamma | {0.1,1,10,100} x {rbf,linear} x {scale,auto} | C=0.1, rbf, scale |
| Logistic Regression | C, solver | {0.1,1,10,100} x {lbfgs,liblinear} | C=0.1, lbfgs |
| Decision Tree | criterion, max_depth, min_samples_split/leaf | {gini,entropy} x {10,20,30,None} x {2,5,10} x {1,2,4} | gini, None, 2, 1 |
| Random Forest | n_estimators, max_depth, min_samples_split/leaf | {100,200,300} x {10,20,30,None} x {2,5} x {1,2} | 100, 10, 2, 1 |
| Gradient Boosting | n_estimators, learning_rate, max_depth | {100,200,300} x {0.05,0.1,0.2} x {3,5,7} | 100, 0.05, 5 |
| XGBoost | n_estimators, learning_rate, max_depth, subsample | {100,200,300} x {0.05,0.1,0.2} x {3,5,7} x {0.8,1.0} | 200, 0.05, 3, 0.8 |

**Table IV: Hyperparameter Search Spaces -- Text-Based Models**

| Model | TF-IDF Params | Classifier Params | Best Configuration |
|---|---|---|---|
| MultinomialNB | max_features:{3K,5K,8K}, ngram:{(1,1),(1,2),(1,3)} | alpha:{0.01,0.1,0.5,1.0} | 8000, (1,2), alpha=0.01 |
| LinearSVC | max_features:{3K,5K,8K}, ngram:{(1,1),(1,2),(1,3)} | C:{0.1,1,10} | 3000, (1,1), C=10 |
| Logistic Regression | max_features:{3K,5K,8K}, ngram:{(1,1),(1,2)} | C:{0.1,1,10,100}, solver:{lbfgs,liblinear} | 3000, (1,1), C=10, liblinear |
| Random Forest | max_features:{3K,5K}, ngram:{(1,1),(1,2)} | n_estimators:{100,200,300}, max_depth:{10,20,None} | 3000, (1,2), 300, None |
| XGBoost | max_features:{3K,5K}, ngram:{(1,1),(1,2)} | n_est:{100,200,300}, lr:{0.05,0.1,0.2}, depth:{3,5,7} | 3000, (1,1), 100, 0.05, 7 |

#### 2) Mathematical Foundations

**Support Vector Machine (SVM):** The SVM finds the optimal separating hyperplane by solving the primal optimization problem:

```
minimize:    (1/2)||w||^2 + C * sum(xi_i)
subject to:  y_i(w^T * phi(x_i) + b) >= 1 - xi_i,   xi_i >= 0
```

With the RBF kernel: `K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)`, the decision function becomes: `f(x) = sign(sum(alpha_i * y_i * K(x_i, x) + b))`. For 41-class classification, the one-vs-one strategy produces 41*(41-1)/2 = 820 binary classifiers.

**Multinomial Naive Bayes:** Using Bayes' theorem with the naive independence assumption:

```
P(y|x_1,...,x_n) proportional to P(y) * product(P(x_i|y))
```

With Laplace smoothing: `P(x_i|y) = (N_yi + alpha) / (N_y + alpha * n)`, where alpha=0.01 provides minimal smoothing, allowing strong feature contributions.

**SMOTE:** Generates synthetic samples along line segments between minority class instances and their k nearest neighbors, effectively expanding the minority class without simple duplication.

### D. Chatbot Architecture

The chatbot employs a 3-layer feedforward neural network implemented in PyTorch:

```
Input (vocab_size) -> Linear + ReLU -> Linear + ReLU -> Linear (num_classes)
```

The NLP preprocessing pipeline consists of: NLTK tokenization -> punctuation removal -> Porter stemming -> bag-of-words encoding. Training uses CrossEntropyLoss with Adam optimizer (lr=0.001) for 1000 epochs with batch size 8. At inference, a softmax confidence threshold of 0.75 filters low-confidence predictions, returning a fallback message for ambiguous queries.

---

## V. System Architecture

### A. Three-Tier Architecture

The system follows a three-tier architecture:

```
+---------------------------------------------------------------+
|                   PRESENTATION TIER                            |
|         HTML5, CSS3, Bootstrap, JavaScript                     |
|         (27 Templates, Responsive UI, Chatbot Widget)          |
+-------------------------------+-------------------------------+
                                | HTTP
+-------------------------------v-------------------------------+
|                     LOGIC TIER                                 |
|                Python Flask (app.py)                            |
|                                                                |
|  +----------+ +----------+ +----------+ +---------+ +-------+ |
|  | Binary   | | Text     | | Diabetes | | Diabetes| | Chat  | |
|  | Predictor| | Predictor| | (PIMA)   | | (BRFSS) | |  Bot  | |
|  | SVM      | | MNB+TFIDF| | SVM      | | RF      | | NN    | |
|  +----------+ +----------+ +----------+ +---------+ +-------+ |
|                                                                |
+-------------------------------+-------------------------------+
                                | SQL / SMTP / File I/O
+-------------------------------v-------------------------------+
|                     DATA TIER                                  |
|   MySQL (pro17)  |  SMTP (Gmail)  |  File System (Reports)    |
+---------------------------------------------------------------+
```

### B. Prediction Flow

**Binary Prediction:** User selects symptoms via checkboxes -> binary vector [0,1]^132 constructed -> SVM model predicts disease + confidence -> specialist mapping from predefined dictionary -> disease information from `disease_info.json`.

**Text Prediction:** User types symptoms in natural language -> TF-IDF pipeline vectorizes text -> MNB predicts top-3 diseases with confidence percentages -> results displayed ranked by probability.

**Diabetes Prediction (PIMA):** User enters 8 clinical parameters -> `StandardScaler` transforms input -> SVM predicts diabetic/non-diabetic.

**Diabetes Prediction (BRFSS):** User enters 21 health indicators -> `StandardScaler` transforms input -> Random Forest predicts risk level + confidence percentage.

### C. Specialist Mapping

The system automatically maps predicted diseases to 8 medical specialist categories: Rheumatologist (2 diseases), Cardiologist (3), ENT Specialist (2), Neurologist (4), Allergist/Immunologist (8), Urologist (2), Dermatologist (5), and Gastroenterologist (8+).

---

## VI. Results and Evaluation

### A. Binary Symptom Model Results

Table V presents the comprehensive results for all six classifiers on the binary symptom dataset.

**Table V: Binary Symptom Model Results (132 features, 41 classes)**

| Model | CV Accuracy (5-fold) | Validation Accuracy | Test Accuracy | Selected |
|---|---|---|---|---|
| **SVM (RBF, C=0.1)** | **100.00%** | **100.00%** | **100.00%** | **Yes** |
| **Logistic Regression (C=0.1)** | **100.00%** | **100.00%** | **100.00%** | |
| Decision Tree (gini) | 100.00% | 100.00% | 97.62% | |
| Random Forest (100 trees) | 100.00% | 100.00% | 97.62% | |
| Gradient Boosting (100 rounds) | 100.00% | 100.00% | 97.62% | |
| XGBoost (200 rounds) | 100.00% | 100.00% | 97.62% | |

SVM was selected as the production model for its 100% accuracy, `predict_proba()` support via Platt scaling, and robust handling of high-dimensional sparse data through the RBF kernel.

### B. Text-Based Model Results

Table VI presents text classification results with optimized TF-IDF configurations.

**Table VI: Text-Based Model Results (TF-IDF vectorization, 24 classes)**

| Model | CV Accuracy | Test Accuracy | Test F1 (weighted) | max_features | ngram_range |
|---|---|---|---|---|---|
| **MultinomialNB** | **96.56%** | **97.92%** | **0.9789** | 8000 | (1,2) |
| LinearSVC | 97.08% | 95.42% | 0.9529 | 3000 | (1,1) |
| Logistic Regression | 97.19% | 95.00% | 0.9488 | 3000 | (1,1) |
| Random Forest | 94.90% | 94.58% | 0.9451 | 3000 | (1,2) |
| XGBoost | 87.40% | 87.08% | 0.8727 | 3000 | (1,1) |

MultinomialNB uniquely benefits from a larger vocabulary (8000 features) and bigram features, while other models performed best with 3000 features, suggesting they are more prone to overfitting on high-dimensional sparse representations.

### C. Diabetes Prediction Results (PIMA)

**Table VII: PIMA Diabetes Model Results (768 samples, 8 features)**

| Model | Test Accuracy |
|---|---|
| **Naive Bayes** | **78.57%** |
| Logistic Regression | 76.62% |
| Random Forest | 75.97% |
| SVM | 75.32% |
| KNN | 75.32% |
| Decision Tree | 73.38% |

Results are consistent with literature benchmarks [12]. The small dataset size (768 records) and limited feature set constrain all models to the 73-79% range.

### D. Diabetes Prediction Results (BRFSS)

**Table VIII: BRFSS Diabetes Model Results (229,474 samples, 21 features, SMOTE-balanced)**

| Model | Test Accuracy |
|---|---|
| **Random Forest (100 trees)** | **84.05%** |
| Gradient Boosting | 83.78% |
| Decision Tree | 77.13% |
| KNN | 75.28% |
| Logistic Regression | 71.50% |
| Naive Bayes | 64.39% |

**Table IX: Classification Report -- BRFSS Random Forest (Best Model)**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Diabetes | 0.87 | 0.96 | 0.91 | 43,667 |
| Diabetes | 0.45 | 0.19 | 0.27 | 11,028 |
| **Overall Accuracy** | | | | **84.05%** |

The confusion matrix reveals:

```
                       Predicted
                  No Diabetes   Diabetes
Actual  No Diabetes   41,857      1,810     (Recall: 96%)
        Diabetes       8,943      2,085     (Recall: 19%)
```

### E. Feature Importance Analysis (BRFSS)

Table X shows the top 10 features ranked by mean decrease in Gini impurity from the Random Forest model.

**Table X: Feature Importance Ranking (BRFSS Random Forest)**

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | BMI | 0.125 | Clinical |
| 2 | Age | 0.098 | Demographic |
| 3 | GenHlth | 0.087 | Self-reported |
| 4 | Income | 0.072 | Socioeconomic |
| 5 | PhysHlth | 0.065 | Self-reported |
| 6 | Education | 0.058 | Socioeconomic |
| 7 | MentHlth | 0.045 | Self-reported |
| 8 | HighBP | 0.042 | Clinical |
| 9 | HighChol | 0.038 | Clinical |
| 10 | Smoker | 0.035 | Behavioral |

BMI emerges as the strongest predictor, consistent with medical literature. Notably, socioeconomic factors (Income, Education) rank in the top 6, highlighting social determinants of health that are absent from traditional clinical-only datasets like PIMA.

### F. Cross-Module Performance Summary

**Table XI: Overall System Performance**

| Module | Best Algorithm | Accuracy | Dataset Size | Feature Dim. |
|---|---|---|---|---|
| Disease (Binary) | SVM (RBF) | **100.00%** | 4,920 | 132 |
| Disease (Text) | TF-IDF + MNB | **97.92%** | 1,200 | 8,000 |
| Diabetes (Basic) | Naive Bayes | **78.57%** | 768 | 8 |
| Diabetes (Advanced) | Random Forest | **84.05%** | 229,474 | 21 |
| Chatbot | PyTorch NN | 0.75 threshold | intents.json | BoW |

---

## VII. Discussion

### A. Why SVM Achieves Perfect Accuracy on Binary Data

The 100% test accuracy of SVM on the binary symptom dataset can be attributed to four factors:

1. **Clean class separation:** Each disease has a unique symptom signature in the 132-dimensional binary feature space. The symptom patterns are sufficiently distinct that no two diseases share identical feature vectors.

2. **High feature-to-class ratio:** With 132 features for 41 classes (~3.2 features per class) and ~120 samples per class, the data creates well-defined clusters in high-dimensional space.

3. **Low regularization effectiveness:** The optimal C=0.1 creates wider margins with more regularization. This prevents overfitting to noise while maintaining perfect separation, as the data is inherently clean.

4. **RBF kernel capacity:** The RBF kernel maps to infinite-dimensional space where any finite dataset is linearly separable. Combined with clean data, this guarantees perfect classification.

However, this result should be interpreted cautiously -- the synthetic nature of the binary symptom dataset (distinct, non-overlapping patterns) does not reflect the complexity of real clinical data where symptom overlap between diseases is common.

### B. Why MNB Outperforms Complex Models on Text Data

MultinomialNB achieves the highest test accuracy (97.92%) despite having a lower CV score (96.56%) than LinearSVC (97.08%) and Logistic Regression (97.19%). This paradox is explained by:

1. **Naive independence as regularization:** The independence assumption acts as implicit regularization, preventing overfitting to spurious word co-occurrences in the small dataset (1,200 samples). More complex models (SVC, LR) overfit these correlations, leading to higher CV but lower test accuracy.

2. **Optimal for sparse high-dimensional data:** TF-IDF produces sparse vectors where most entries are zero. MNB naturally handles sparse count-based features, while tree-based models (RF, XGBoost) struggle with sparsity.

3. **Bigram synergy:** MNB uniquely benefits from max_features=8000 with ngram=(1,2), capturing medical bigrams like "joint pain", "skin rash", and "chest tightness". Other models performed best with only 3000 features, suggesting information overload at higher dimensions.

4. **Low smoothing leverage:** Alpha=0.01 allows the model to trust the data strongly. With 50 representative samples per class and clear medical vocabulary, minimal smoothing is optimal.

### C. SMOTE Impact on Diabetes Prediction

The BRFSS dataset presents a 6.2:1 class imbalance (No Diabetes vs. Diabetes). Without SMOTE, a naive classifier achieves ~86% accuracy by always predicting the majority class, while missing most diabetic patients. SMOTE balances the training set (50/50), resulting in:

- **Overall accuracy:** Decreases from ~86% to 84.05% (acceptable trade-off)
- **Diabetes recall:** The recall of 0.19 indicates that even with SMOTE, the model misses approximately 81% of diabetic patients -- a significant clinical limitation
- **No Diabetes recall:** High at 0.96, meaning very few healthy patients are incorrectly flagged

For clinical screening, recall is prioritized over precision -- it is preferable to have false positives (additional testing) than false negatives (missed diagnoses). The low diabetes recall suggests that behavioral and demographic features alone, while informative, are insufficient for robust diabetes screening without clinical biomarkers.

### D. Algorithm Selection is Task-Dependent

Our results demonstrate that no single algorithm dominates across all tasks:

- **SVM** excels on high-dimensional, cleanly separated binary data
- **MNB** excels on sparse, high-dimensional text data with small sample sizes
- **Random Forest** excels on large-scale, mixed-type data with complex interactions
- **Naive Bayes** (Gaussian) excels on small clinical datasets with independent features

This finding reinforces the importance of comprehensive benchmarking rather than defaulting to a single "best" algorithm.

### E. Comparison with Existing Systems

**Table XII: Comparison with Existing Systems**

| Feature | Traditional Systems | Existing ML Systems | **Our System** |
|---|---|---|---|
| Prediction Modes | 0 (manual) | 1 | **4** |
| Input Modality | Physical exam | Structured only | **Structured + NLP** |
| Diabetes Features | Lab tests | 8 (PIMA only) | **8 + 21 (dual-scale)** |
| Dataset Scale | N/A | < 1,000 | **Up to 229,474** |
| Specialist Mapping | Manual referral | Not available | **Automatic (8 specialties)** |
| Appointment System | Phone/walk-in | Not available | **Online + Payment** |
| Chatbot Support | None | Basic FAQ | **PyTorch NN** |
| LLM Integration | None | None | **Gemma 3 (Ollama)** |

### F. Limitations

1. **Dataset limitations:** The binary symptom test set contains only 42 samples (1 per disease). The 100% accuracy, while valid on this test set, requires validation on larger, real-world clinical data with overlapping symptom patterns.

2. **Low diabetes recall:** The BRFSS model's 19% recall for the diabetes class means approximately 81% of actual diabetic patients would be missed -- a critical limitation for any clinical screening tool.

3. **Feature limitations:** The binary model lacks severity levels (symptoms are binary), temporal information (duration/progression), patient demographics, and medical history.

4. **Dataset bias:** PIMA is limited to Pima Indian women aged 21+. BRFSS is US-based. Symptom2Disease is English-only. These biases limit generalizability.

5. **No clinical validation:** The system has not been validated in a clinical setting with real patient populations.

6. **Static models:** Models are pre-trained and do not update with new data. No online learning or feedback loop exists.

---

## VIII. Disease Information Integration via LLM

### A. Gemma 3 Integration

The `generate_disease_info.py` module leverages Gemma 3 (a lightweight open-source LLM by Google) running locally via Ollama (`http://localhost:11434/api/generate`) to generate structured disease information for all 24 text-predicted diseases.

### B. Generation Pipeline

For each disease:
1. Retrieve 5 sample patient descriptions from the Symptom2Disease dataset
2. Construct a structured prompt requesting JSON-formatted medical information
3. Send to Gemma 3 via Ollama API
4. Parse JSON response with boundary detection (`raw.index("{")`, `raw.rindex("}"))`)
5. Retry with simplified prompt on parse failure
6. Save to `disease_info.json`

### C. Generated Information Fields

Each disease entry contains:
- **description:** 2-3 sentence clinical overview
- **common_symptoms:** List of 5 characteristic symptoms
- **precautions:** List of 4 preventive measures
- **medications:** List of 3 medication categories
- **diet_recommendations:** List of 3 dietary guidelines
- **when_to_see_doctor:** Urgency criteria and warning signs

This LLM-generated content enriches prediction results with actionable medical information, enhancing the clinical utility of the system beyond simple disease labels.

---

## IX. Implementation Details

### A. Technology Stack

**Table XIII: Technology Stack**

| Layer | Technology | Purpose |
|---|---|---|
| Web Framework | Flask | Routing, request handling, session management |
| ML Core | scikit-learn | Classification, preprocessing, evaluation |
| Boosting | XGBoost | Gradient boosted classification |
| Deep Learning | PyTorch | Neural network chatbot |
| NLP | NLTK | Tokenization, stemming, bag-of-words |
| Data Processing | pandas, NumPy | Data manipulation and computation |
| Class Balancing | imbalanced-learn | SMOTE implementation |
| Visualization | matplotlib, seaborn | Correlation heatmaps, analysis plots |
| Model Persistence | joblib, pickle | Model serialization/deserialization |
| Payment | Razorpay | PCI-DSS compliant payment processing |
| Email | SMTP (Gmail) | Patient notification delivery |
| LLM | Ollama (Gemma 3) | Disease information generation |
| Database | MySQL (Flask-MySQLdb) | User, doctor, appointment management |
| Frontend | HTML5/CSS3/Bootstrap/JS | Responsive UI with 27 templates |

### B. Model Serialization

| Model File | Format | Content |
|---|---|---|
| `best_binary_model.joblib` | Joblib | SVM (100% accuracy) |
| `best_text_model.joblib` | Joblib | TF-IDF + MNB pipeline (97.92%) |
| `binary_label_encoder.joblib` | Joblib | Disease label encoder (41 classes) |
| `text_label_encoder.joblib` | Joblib | Disease label encoder (24 classes) |
| `svm_model.pkl` | Pickle | PIMA diabetes SVM |
| `scaler.pkl` | Pickle | PIMA StandardScaler |
| `diabetes_brfss_model.pkl` | Pickle | BRFSS Random Forest (84.05%) |
| `diabetes_brfss_scaler.pkl` | Pickle | BRFSS StandardScaler |
| `diabetes_brfss_features.pkl` | Pickle | BRFSS feature names (21) |
| `data.pth` | PyTorch | Chatbot NN state dict |

All models are loaded once at Flask application startup and stored as global variables, ensuring efficient memory usage and fast inference (< 10ms per prediction).

### C. Web Application Routes

The Flask application (877 lines) implements 25+ routes organized by role:

- **User routes (17):** Registration, login, 4 prediction endpoints, appointment booking, payment flow, history
- **Admin routes (7):** Dashboard, user CRUD, doctor CRUD
- **Doctor routes (5):** Dashboard, appointment management, email notifications

### D. Security Measures

- SQL injection prevention via parameterized queries (`%s` placeholders)
- File upload sanitization via Werkzeug `secure_filename()`
- Session-based authentication with Flask secret key
- PCI-DSS compliant payment processing via Razorpay (no card data stored)
- Role-based access control (User, Doctor, Admin)

---

## X. Conclusion

### A. Key Findings

This paper presented a comprehensive multi-modal disease prediction system integrating four ML modules into a unified web-based platform. The key findings are:

1. **SVM with RBF kernel** (C=0.1, gamma=scale) achieves perfect 100% accuracy on binary symptom classification across 41 diseases, attributed to the clean separability of the dataset's symptom signatures in 132-dimensional space.

2. **Multinomial Naive Bayes with TF-IDF** (max_features=8000, ngram=(1,2), alpha=0.01) achieves 97.92% accuracy on text-based classification across 24 diseases, outperforming more complex models (LinearSVC, Logistic Regression, Random Forest, XGBoost) due to the naive independence assumption acting as natural regularization on the small dataset.

3. **SMOTE effectively balances** the BRFSS diabetes dataset (229,474 records), though the minority class recall remains low (0.19), indicating that behavioral health indicators alone are insufficient for robust diabetes screening.

4. **Algorithm selection is task-dependent:** SVM excels on high-dimensional binary data, MNB on sparse text data, Random Forest on large-scale mixed-type data, and Gaussian Naive Bayes on small clinical datasets.

5. **Multi-modal integration** combining binary symptoms, free-text NLP, basic clinical parameters, and population-level health indicators provides a versatile, accessible health screening tool that addresses the limitations of single-modality systems.

6. **LLM integration** (Gemma 3 via Ollama) demonstrates the practical application of large language models for generating structured medical content that enriches prediction results.

### B. Contributions

1. **Comprehensive benchmarking** of 6 classifiers across 4 datasets with full GridSearchCV hyperparameter optimization.
2. **Dual-modality disease prediction** combining structured binary inputs and free-text NLP in a single system.
3. **Dual-scale diabetes assessment** integrating clinical features (PIMA) with population-level behavioral indicators (BRFSS).
4. **Integrated clinical workflow** from prediction through specialist mapping, appointment booking, payment processing, and email notifications.
5. **LLM-enriched knowledge base** using Gemma 3 for automated disease information generation.

### C. Clinical Implications

The system is intended as a **preliminary screening tool** and not a replacement for professional medical diagnosis. The high accuracy on binary symptom prediction (100%) and text prediction (97.92%) demonstrates the feasibility of ML-based preliminary screening. However, the low diabetes recall (19%) underscores the need for clinical biomarkers alongside behavioral indicators. All predictions include confidence scores to help users assess reliability.

### D. Future Work

**Short-term:** Expand disease coverage to 100+ diseases; add patient demographics (age, sex) as features; implement SHAP/LIME for model interpretability; conduct clinical validation with real patient data.

**Medium-term:** Explore deep learning for text classification (BioBERT, ClinicalBERT); implement temporal modeling of symptom progression using RNNs/LSTMs; add multi-label prediction for comorbidity detection; deploy active learning for continuous model improvement.

**Long-term:** Integrate with Electronic Health Record (EHR) systems; implement federated learning for privacy-preserving model training across institutions; add multi-language support for global accessibility; explore regulatory approval pathways.

---

## Acknowledgments

The authors thank the developers of scikit-learn [5], PyTorch [9], Flask, NLTK, and Ollama for their open-source contributions. We also acknowledge the creators of the publicly available datasets: the PIMA Diabetes Dataset (National Institute of Diabetes and Digestive and Kidney Diseases), the CDC BRFSS 2015 dataset, the Symptom2Disease dataset, and the Binary Symptom-Disease dataset used in this research.

---

## References

[1] Z. Obermeyer and E. J. Emanuel, "Predicting the future -- Big data, machine learning, and clinical medicine," *New England Journal of Medicine*, vol. 375, no. 13, pp. 1216-1219, 2016.

[2] Y. Deepthi, K. P. Kalyan, M. Vyas, K. Radhika, D. K. Babu, and N. V. Krishna Rao, "Disease prediction based on symptoms using machine learning," in *Energy Systems, Drives and Automations: Proceedings of ESDA 2019*, Springer, 2020.

[3] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," in *Proc. Annual Symp. on Computer Application in Medical Care*, pp. 261-265, 1988.

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321-357, 2002.

[5] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, pp. 785-794, 2016.

[7] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995.

[8] Centers for Disease Control and Prevention, "Behavioral Risk Factor Surveillance System (BRFSS)," 2015. [Online]. Available: https://www.cdc.gov/brfss/

[9] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, pp. 8024-8035, 2019.

[10] A. Kunjir, H. Sawant, and N. Shaikh, "Data mining and visualization for prediction of multiple diseases in healthcare," in *Proc. Int. Conf. on Big Data Analytics and Computational Intelligence (ICBDACI)*, pp. 329-334, 2017.

[11] S. Wang and C. D. Manning, "Baselines and bigrams: Simple, good sentiment and topic classification," in *Proc. 50th Annual Meeting of the ACL*, pp. 90-94, 2012.

[12] D. Sisodia and D. Sisodia, "Prediction of diabetes using classification algorithms," *International Journal of Computer Applications*, vol. 176, no. 4, pp. 19-23, 2018.

[13] A. K. Singh, S. Kumar, M. Singh et al., "Machine learning approach for diabetes prediction using BRFSS dataset," *Procedia Computer Science*, 2022.

[14] A. Hamdi, M. Mohamed, R. Emad, and K. Shaban, "An ensemble classification approach in a multi-layered large language model framework for disease prediction," *arXiv preprint arXiv:2509.02446*, Sep. 2025.

[15] R. Zannat, A. Al Shafi, and A. Muntakim, "Bridging the gap in Bangla healthcare: Machine learning based disease prediction using a symptoms-disease dataset," *IEEE ECCE 2025* (arXiv:2601.12068), Jan. 2026.

[16] A. Fernandez et al., "SMOTE for learning from imbalanced data: Progress and challenges," *Journal of Artificial Intelligence Research*, vol. 61, pp. 863-905, 2018.

[17] A. McCallum and K. Nigam, "A comparison of event models for Naive Bayes text classification," in *AAAI-98 Workshop on Learning for Text Categorization*, 1998.

[18] S. Bird, E. Loper, and E. Klein, *Natural Language Processing with Python*, O'Reilly Media Inc., 2009.

[19] M. Pal, "Random forest classifier for remote sensing classification," *International Journal of Remote Sensing*, vol. 26, no. 1, pp. 217-222, 2005.

[20] W. McKinney, "Data structures for statistical computing in Python," in *Proc. 9th Python in Science Conference*, pp. 56-61, 2010.

---

## Appendix A: Complete Hyperparameter Results

### A.1 Binary Models -- Detailed Results

| Model | CV Score | Val Acc | Test Acc | Best C | Best Kernel | Other Params |
|---|---|---|---|---|---|---|
| SVM | 1.0000 | 1.0000 | 1.0000 | 0.1 | rbf | gamma=scale |
| Logistic Reg. | 1.0000 | 1.0000 | 1.0000 | 0.1 | -- | solver=lbfgs, max_iter=2000 |
| Decision Tree | 1.0000 | 1.0000 | 0.9762 | -- | -- | gini, depth=None, split=2, leaf=1 |
| Random Forest | 1.0000 | 1.0000 | 0.9762 | -- | -- | 100 trees, depth=10, split=2, leaf=1 |
| Gradient Boost | 1.0000 | 1.0000 | 0.9762 | -- | -- | 100 rounds, lr=0.05, depth=5 |
| XGBoost | 1.0000 | 1.0000 | 0.9762 | -- | -- | 200 rounds, lr=0.05, depth=3, sub=0.8 |

### A.2 Text Models -- Detailed Results

| Model | CV Score | Test Acc | F1 | max_features | ngram | Classifier Params |
|---|---|---|---|---|---|---|
| MultinomialNB | 0.9656 | 0.9792 | 0.9789 | 8000 | (1,2) | alpha=0.01 |
| LinearSVC | 0.9708 | 0.9542 | 0.9529 | 3000 | (1,1) | C=10 |
| Logistic Reg. | 0.9719 | 0.9500 | 0.9488 | 3000 | (1,1) | C=10, liblinear |
| Random Forest | 0.9490 | 0.9458 | 0.9451 | 3000 | (1,2) | 300 trees, depth=None |
| XGBoost | 0.8740 | 0.8708 | 0.8727 | 3000 | (1,1) | 100 rounds, lr=0.05, depth=7 |

### A.3 BRFSS Models -- Detailed Results

| Model | Test Accuracy | Configuration |
|---|---|---|
| Random Forest | 84.05% | n_estimators=100, n_jobs=-1, random_state=42 |
| Gradient Boosting | 83.78% | n_estimators=100, random_state=42 |
| Decision Tree | 77.13% | random_state=42 |
| KNN | 75.28% | k=5 (default) |
| Logistic Regression | 71.50% | max_iter=1000, solver=liblinear |
| Naive Bayes | 64.39% | Gaussian (default) |

---

## Appendix B: Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Overall correct prediction rate |
| Precision | TP / (TP+FP) | Positive predictive value |
| Recall | TP / (TP+FN) | Sensitivity / True positive rate |
| F1-Score | 2*(P*R) / (P+R) | Harmonic mean of precision and recall |
| CV Score | Mean accuracy across k folds | Robust generalization estimate |

---

*Disclaimer: This system is intended for educational and preliminary screening purposes only. It is not a substitute for professional medical diagnosis. Users should consult qualified healthcare providers for any health concerns.*

---

<p align="center"><b>End of Paper</b></p>
