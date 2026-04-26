# Disease Prediction Based on Symptoms - PPT Presentation Guide

---

## Slide 1: Title Slide
- **Project Title:** Disease Prediction Based on Symptoms
- **Tagline:** An AI-powered Healthcare Web Application for Early Disease Detection
- **Your Name / Team Members / Guide Name / University Logo**

---

## Slide 2: Problem Statement
- Manual disease diagnosis is time-consuming and prone to human error
- Patients often do not know which specialist to consult
- Lack of accessible, automated preliminary diagnosis tools
- Delayed diagnosis leads to worsening health conditions
- Need for a system that can predict diseases from symptoms quickly and accurately

---

## Slide 3: Objectives
- Develop a web-based disease prediction system using Machine Learning
- Predict diseases from user-selected symptoms with confidence scores
- Predict diseases from free-text symptom descriptions using NLP
- Provide a Diabetes risk assessment module (Basic: 8 features, Advanced: 21 features with 84% accuracy)
- Build an AI-powered Chatbot for health-related queries
- Enable online appointment booking with specialist doctors
- Implement payment gateway integration (Razorpay)
- Manage users, doctors, and appointments through Admin/Doctor/User panels

---

## Slide 4: System Architecture / High-Level Design
```
User Browser (HTML/CSS/JS)
        |
    Flask Web Server (app.py)
        |
   +----+----+----+----+
   |    |    |    |    |
 MySQL  ML   NLP  Chat Payment
Database Models Models Bot Gateway
```
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Backend:** Python Flask
- **Database:** MySQL
- **ML Models:** Scikit-learn (Random Forest, Decision Tree, Gradient Boosting, Naive Bayes, SVM)
- **Deep Learning:** PyTorch Neural Network (Chatbot)
- **NLP:** NLTK (Tokenization, Stemming, Bag of Words)
- **Payment:** Razorpay API
- **Email:** SMTP (Gmail)

---

## Slide 4a: Data Flow Diagram (DFD)

### Level 0 DFD (Context Diagram)
```
                    ┌─────────────────────┐
   ┌───────────┐    │                     │    ┌───────────┐
   │           │───▶│                     │───▶│           │
   │   User    │    │   Disease Prediction│    │  Doctor   │
   │  (Patient)│◀───│    Web Application  │◀───│   Panel   │
   │           │    │                     │    │           │
   └───────────┘    │                     │    └───────────┘
                    │   ┌───────────┐     │
   ┌───────────┐    │   │  MySQL DB │     │    ┌───────────┐
   │           │───▶│   │  (pro17)  │     │───▶│           │
   │   Admin   │    │   └───────────┘     │    │  Razorpay │
   │   Panel   │◀───│                     │◀───│  Payment  │
   │           │    │   ┌───────────┐     │    └───────────┘
   └───────────┘    │   │ML Models  │     │
                    │   │+ Chatbot  │     │
                    │   └───────────┘     │
                    └─────────────────────┘
```

### Level 1 DFD
```
User ──▶ [1.0 Auth] ──▶ User Session
                │
                ▼
User ──▶ [2.0 Disease Pred] ──▶ ML Model ──▶ Result + Specialist
                │
                ▼
User ──▶ [3.0 Diabetes Pred] ──▶ ML Model ──▶ Risk Score
                │
                ▼
User ──▶ [4.0 Book Appt] ──▶ [5.0 Payment] ──▶ DB ──▶ Doctor
                │
                ▼
User ──▶ [6.0 Chatbot] ──▶ PyTorch NN ──▶ Response

Admin ─▶ [7.0 Manage Users/Doctors] ──▶ MySQL DB
Doctor ─▶ [8.0 Manage Appointments] ──▶ MySQL DB ──▶ Email (SMTP)
```

---

## Slide 4b: Use Case Diagram (UML)
```
                        ┌──────────────────────────────────────────┐
                        │       Disease Prediction System          │
                        │                                          │
   ┌──────┐    ┌──────┐ │  ┌──────────┐  ┌──────────────┐        │
   │Login │    │Regis│ │  │ Predict   │  │ Text-Based   │        │
   │      │◀───│ter   │ │  │ Disease   │  │ Prediction   │        │
   └──┬───┘    └──┬───┘ │  │ (Binary)  │  │ (NLP)        │        │
      │           │     │  └─────┬─────┘  └──────┬───────┘        │
      │           │     │        │               │                │
   ┌──▼───────────▼──┐  │  ┌─────▼─────┐  ┌──────▼───────┐        │
   │                  │  │  │ Diabetes  │  │  Diabetes    │        │
   │     USER         │◀─│  │ Basic     │  │  Advanced    │        │
   │  (Patient)       │  │  │ (8 feat)  │  │  (21 feat)   │        │
   │                  │  │  └───────────┘  └──────────────┘        │
   └──┬───┬───┬───┬───┘  │                                    ┌────┴───┐
      │   │   │   │      │  ┌───────────┐  ┌──────────┐      │Chat-   │
      │   │   │   │      │  │ Book      │  │ View     │      │bot     │
      │   │   │   └──────│─▶│ Appt      │  │ History  │      │(AI)    │
      │   │   │          │  └─────┬─────┘  └──────────┘      └────────┘
      │   │   └──────────│───────│─────────▶                    ▲
      │   │              │       ▼                               │
      │   │        ┌─────┴──────┐                               │
      │   │        │  Razorpay  │    ┌────────────────┐         │
      │   │        │  Payment   │    │ Manage Appts   │◀────────┤
      │   │        └────────────┘    │ Accept/Reject  │         │
      │   │                          │ Send Email     │    ┌────┴───┐
      │   │                          └───────┬────────┘    │DOCTOR  │
      │   │                                  │             └────────┘
      │   │                          ┌───────▼────────┐
      │   │                          │ Manage Users   │    ┌────┐
      │   └──────────────────────────│ Manage Doctors │◀───│ADMIN│
      │                              │ Dashboard      │    └────┘
      └──────────────────────────────└────────────────┘
```

---

## Slide 5: Database Design
- **Database:** MySQL (Database Name: `pro17`)
- **Tables:**
  1. **admin** - Stores admin credentials (id, username, password)
  2. **user** - Stores patient information (u_id, u_name, u_email, u_password, u_mobile, u_age)
  3. **doctors** - Stores doctor information (d_id, d_name, d_email, d_passwords, d_spec)
  4. **appointment** - Stores appointment records (ap_id, d_id, u_id, ap_time, ap_date, ap_report, ap_payment_status, ap_status, razorpay_order_id)
- **Relationships:** Foreign keys from appointment to doctors and user tables (ON DELETE CASCADE)

---

## Slide 6: Dataset Details
### Dataset 1: Disease Symptom Dataset
- **Training Data:** `dataset/training_data.csv`
- **Test Data:** `dataset/test_data.csv`
- **Features:** 132 binary symptom columns (0 or 1)
- **Target:** `prognosis` column (41 diseases)
- **Diseases Covered (41):** Fungal infection, Allergy, GERD, Diabetes, Hypertension, Migraine, Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A-E, Tuberculosis, Common Cold, Pneumonia, Heart attack, Arthritis, Acne, Psoriasis, Impetigo, and more
- **Symptoms Covered (132):** itching, skin_rash, headache, fever, cough, vomiting, fatigue, joint_pain, chest_pain, nausea, dizziness, and more

### Dataset 2: Diabetes Dataset (Basic)
- **Source:** PIMA Diabetes Dataset (`diabetes.csv`)
- **Records:** 768 patients
- **Features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (0 = Non-Diabetic, 1 = Diabetic)

### Dataset 3: Diabetes Dataset (Advanced - BRFSS 2015)
- **Source:** CDC BRFSS 2015 (`diabetes_binary_health_indicators_BRFSS2015.csv`)
- **Records:** 253,680 patients (229,474 after deduplication)
- **Features (21):** HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
- **Target:** Diabetes_binary (0 = No Diabetes, 1 = Diabetes)
- **Class Distribution:** No Diabetes: 218,334 | Diabetes: 35,346
- **Preprocessing:** SMOTE applied to handle class imbalance
- **Train/Test Split:** 80/20 (stratified)
- **Duplicate Rows Removed:** 24,206

### Dataset 3: Symptom2Disease (Text-based)
- **File:** `Symptom2Disease.csv`
- **Records:** 1,200 patient symptom descriptions
- **Columns:** Unnamed: 0 (index), label (disease name), text (natural language symptom description)
- **Diseases Covered (24):** Psoriasis, Varicose Veins, Typhoid, Chicken pox, Impetigo, Dengue, Fungal infection, Common Cold, Pneumonia, Dimorphic Hemorrhoids, Arthritis, Acne, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Jaundice, Malaria, Urinary tract infection, Allergy, GERD, Drug reaction, Peptic ulcer disease, Diabetes
- **Samples per Disease:** 50 descriptions each (balanced dataset)
- **Example Text:** *"I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."* -> **Psoriasis**
- **Used for:** Text-based disease prediction model (`best_text_model.joblib`) using NLP
- **Also used with:** Gemma 3 (LLM via Ollama) in `generate_disease_info.py` to generate structured disease information (description, common_symptoms, precautions, medications, diet_recommendations, when_to_see_doctor)

---

## Slide 7: Literature Review

### Study 1: Disease Prediction from Symptoms Using ML
- **Authors:** Y. Deepthi, K.P. Kalyan et al. (2020, Springer)
- **Finding:** Compared Naive Bayes, Decision Tree, and Random Forest for symptom-based disease prediction. Decision Tree and Random Forest achieved >95% accuracy on structured symptom datasets.
- **Relevance:** Our project uses the same symptom-disease classification approach with Decision Tree (100%), Naive Bayes (100%), and SVM (100%) on 132 symptoms → 41 diseases.

### Study 2: ML-Based Diabetes Prediction on PIMA Dataset
- **Authors:** D. Sisodia, D. Sisodia (2018, International Journal of Computer Applications)
- **Finding:** Compared Logistic Regression (78.3%), Decision Tree (73.4%), Naive Bayes (76.3%), SVM (77.6%), and KNN (72.4%) on the PIMA Diabetes Dataset. SVM and Logistic Regression performed best.
- **Relevance:** We used the same PIMA dataset and achieved comparable results — Naive Bayes 78.57%, Logistic Regression 76.62%, SVM 75.32%.

### Study 3: Diabetes Prediction Using BRFSS Data
- **Authors:** A. K. Singh, S. Kumar et al. (2022, Journal of Diabetes Research)
- **Finding:** Used CDC BRFSS health survey data with 21+ features. Random Forest and Gradient Boosting achieved 80-85% accuracy, significantly outperforming single-feature models.
- **Relevance:** Our BRFSS model achieved 84.05% with Random Forest using 21 health indicators from 229,474 records.

### Study 4: NLP-Based Disease Prediction from Symptom Text
- **Authors:** A. Hamdi, M. Mohamed et al. (2025, arXiv:2509.02446)
- **Finding:** Used fine-tuned transformer models (BERT-based) with ensemble learning for disease classification from symptom text. Achieved 80.56% accuracy using majority voting ensemble.
- **Relevance:** Our text-based pipeline achieves 97.92% accuracy using TF-IDF + classifier on the Symptom2Disease dataset (1,200 records, 24 diseases).

### Study 5: Ensemble Approaches for Disease Prediction
- **Authors:** R. Zannat, A. Al Shafi et al. (2026, IEEE ECCE)
- **Finding:** Ensemble voting approaches with 758 symptom-disease relationships across 85 diseases achieved 98% accuracy.
- **Relevance:** Our project uses ensemble and individual models; Decision Tree, Naive Bayes, and SVM all achieve 100% on the binary symptom dataset.

### Key Literature Gaps Addressed by Our Project:
1. Most studies focus on **single prediction mode** (symptoms OR diabetes). We integrate **4 prediction modes** in one application.
2. Limited studies use **21+ health indicators** for diabetes prediction — we leverage BRFSS with 21 features.
3. Few systems provide **end-to-end functionality** (prediction → specialist mapping → appointment booking → payment).

---

## Slide 8: Machine Learning Models & Accuracy Results

### Model 1: Disease Prediction (Binary Symptoms) — 132 features, 41 diseases
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `best_binary_model.joblib` | SVM (RBF kernel, C=0.1) | **100.00%** |
| `decision_tree.joblib` | Decision Tree (gini) | **100.00%** |
| `mnb.joblib` | Multinomial Naive Bayes | **100.00%** |

- **Why 100%?** The symptom-disease dataset has distinct, non-overlapping symptom patterns per disease
- **Validation Split:** 33% test size, 3-fold cross-validation
- **Production Model:** SVM (best_binary_model) — chosen for probability estimates and robustness

### Model 2: Disease Prediction (Text-Based) — Natural Language Input
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `best_text_model.joblib` | Scikit-learn Pipeline (TF-IDF + MNB) | **97.92%** |

- **Dataset:** Symptom2Disease — 1,200 text descriptions across 24 diseases
- **Output:** Top-3 predicted diseases with confidence scores
- **Best Params:** max_features=8000, ngram_range=(1,2), alpha=0.01

### Model 3: Diabetes Prediction (Basic - PIMA) — 8 features, 768 records
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `svm_model.pkl` | **Support Vector Machine** | 75.32% |
| `scaler.pkl` + retrained | **Naive Bayes** | **78.57% (Best)** |
| `scaler.pkl` + retrained | **Logistic Regression** | 76.62% |
| `scaler.pkl` + retrained | **Random Forest** | 75.97% |
| `scaler.pkl` + retrained | **KNN** | 75.32% |
| `scaler.pkl` + retrained | **Decision Tree** | 73.38% |

### Model 4: Diabetes Prediction (Advanced - BRFSS 2015) — 21 features, 229,474 records
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `diabetes_brfss_model.pkl` | **Random Forest (100 estimators)** | **84.05% (Best)** |
| Tested | **Gradient Boosting** | 83.78% |
| Tested | **Decision Tree** | 77.13% |
| Tested | **KNN** | 75.28% |
| Tested | **Logistic Regression** | 71.50% |
| Tested | **Naive Bayes** | 64.39% |

### Model 5: Chatbot (Deep Learning)
| Model | Framework | Type |
|-------|-----------|------|
| `data.pth` | PyTorch | 3-layer Feedforward Neural Network |

### Overall Accuracy Summary Table
| Module | Best Algorithm | Accuracy | Dataset Size |
|--------|---------------|----------|-------------|
| Disease Prediction (Binary) | SVM / Decision Tree / MNB | **100.00%** | 4,920 train + test |
| Disease Prediction (Text) | TF-IDF + MNB | **97.92%** | 1,200 records |
| Diabetes (Basic - PIMA) | Naive Bayes / SVM | **78.57%** | 768 records |
| Diabetes (Advanced - BRFSS) | Random Forest | **84.05%** | 229,474 records |
| Chatbot | PyTorch Neural Network | 0.75 threshold | intents.json |

---

## Slide 8a: Mathematical Formulation of Key Algorithms

### Support Vector Machine (SVM)
- **Objective:** Find the hyperplane that maximizes the margin between classes
- **Decision function:** f(x) = w^T · φ(x) + b
- **Optimization:** min (1/2)||w||² + C Σ ξᵢ subject to yᵢ(w^T · φ(xᵢ) + b) ≥ 1 - ξᵢ
- **RBF Kernel:** K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)
- **Parameters used:** C=0.1, kernel=rbf, gamma=scale

### Random Forest
- **Ensemble prediction:** ĥ(x) = (1/B) Σ ᵢ₌₁ᴮ hᵢ(x) (majority vote)
- **Each tree trained on bootstrap sample with random feature subset**
- **Gini impurity:** Gini(t) = 1 - Σ p(i|t)²
- **Parameters used:** n_estimators=100, max_depth=None, random_state=42

### Multinomial Naive Bayes (Text Model)
- **Bayes theorem:** P(y|x) = P(x|y) · P(y) / P(x)
- **With independence assumption:** P(y|x₁,...,xₙ) ∝ P(y) Πᵢ P(xᵢ|y)
- **Multinomial:** P(xᵢ|y) = (N_yᵢ + α) / (N_y + α·n)
- **Parameters used:** alpha=0.01 (Laplace smoothing)

### TF-IDF Vectorization
- **TF(t,d)** = (Number of times term t appears in document d) / (Total terms in d)
- **IDF(t)** = log((1 + n) / (1 + df(t))) + 1
- **TF-IDF(t,d)** = TF(t,d) × IDF(t)
- **Parameters used:** max_features=8000, ngram_range=(1,2)

### Neural Network (Chatbot)
- **Forward pass:** z₁ = W₁·x + b₁ → a₁ = ReLU(z₁) → z₂ = W₂·a₁ + b₂ → a₂ = ReLU(z₂) → z₃ = W₃·a₂ + b₃
- **Loss:** CrossEntropyLoss = -Σ cᵢ log(pᵢ)
- **Architecture:** Input → Hidden(ReLU) → Hidden(ReLU) → Output

---

## Slide 9: Deep Learning - Chatbot (Neural Network)
- **Framework:** PyTorch
- **Architecture:**
  - Input Layer: `input_size` neurons
  - Hidden Layer 1: `hidden_size` neurons + ReLU activation
  - Hidden Layer 2: `hidden_size` neurons + ReLU activation
  - Output Layer: `num_classes` neurons
- **NLP Pipeline (nltk_utils.py):**
  - Tokenization using NLTK `word_tokenize`
  - Stemming using PorterStemmer
  - Bag of Words representation
- **Training Data:** `intents.json` with intents for greeting, goodbye, thanks, services, payments, symptom descriptions, disease info queries
- **Model File:** `data.pth` (saved PyTorch state dict)
- **Confidence Threshold:** 0.75 (responses below threshold: "I do not understand...")
- **Bot Name:** Sam

---

## Slide 10: Application Modules / Features

### User Panel
- User Registration & Login
- Disease Prediction (Binary Symptom Selection)
- Disease Prediction (Text-Based Symptom Input)
- Diabetes Risk Assessment (Basic - 8 parameters)
- Diabetes Risk Assessment (Advanced - 21 health indicators, 84% accuracy)
- Book Appointment with Doctors
- Upload Medical Reports
- Make Payment via Razorpay
- View Appointment History
- Chatbot for Health Queries

### Doctor Panel
- Doctor Login
- View Appointments
- Accept / Reject Appointments
- Send Email Reminders to Patients

### Admin Panel
- Admin Login
- Dashboard (Total Users, Total Doctors)
- Manage Users (Add, Edit, Delete)
- Manage Doctors (Add, Delete)

### AI Chatbot
- Floating chat widget on the website
- Responds to greetings, service queries, symptom descriptions
- Provides disease information

---

## Slide 11: Technology Stack
| Category | Technology |
|----------|-----------|
| Frontend | HTML5, CSS3, Bootstrap, JavaScript |
| Backend | Python Flask |
| Database | MySQL (via Flask-MySQLdb) |
| Machine Learning | Scikit-learn (Random Forest, Decision Tree, Gradient Boosting, SVM, Naive Bayes, KNN, Logistic Regression) |
| Deep Learning | PyTorch |
| NLP | NLTK (Tokenization, Stemming, Bag of Words) |
| Data Processing | Pandas, NumPy |
| Class Balancing | imbalanced-learn (SMOTE) |
| Data Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib, Pickle |
| Payment Gateway | Razorpay |
| Email Service | SMTP (Gmail) |
| File Upload | Werkzeug |
| LLM Integration | Ollama (Gemma 3) for disease info generation |
| Conda Environment | Anaconda (environment.yml) |

---

## Slide 12: Comparison with Existing Systems
| Feature | Traditional Diagnosis | Existing ML Systems | Our System |
|---------|----------------------|---------------------|------------|
| Disease Prediction | Manual only | Single mode (symptoms only) | **4 modes** (Binary, Text, Diabetes Basic, Diabetes Advanced) |
| Input Method | Physical consultation | Structured form only | Structured + Natural Language text |
| Diabetes Assessment | Lab tests only | Basic features (8) | Basic (8) + Advanced (21 health indicators) |
| Specialist Mapping | Manual referral | Not available | **Automatic** specialist recommendation |
| Appointment Booking | Phone/Walk-in | Not available | **Online** with payment gateway |
| Chatbot Support | Not available | Basic FAQ | **AI-powered** health query assistant |
| Multi-role Access | Not applicable | Single user type | **3 roles** (User, Doctor, Admin) |
| Dataset Scale | N/A | <1000 records | **Up to 229,474 records** (BRFSS) |
| Accuracy (Disease) | Variable | 85-95% | **100%** (Binary), **97.92%** (Text) |
| Accuracy (Diabetes) | N/A | 75-80% | **84.05%** (Advanced BRFSS) |

---

## Slide 13: Challenges Faced & Solutions

### Challenge 1: Class Imbalance in BRFSS Diabetes Dataset
- **Problem:** No Diabetes: 218,334 vs Diabetes: 35,346 (heavily imbalanced)
- **Solution:** Applied SMOTE (Synthetic Minority Oversampling Technique) to balance training data, improving minority class recall

### Challenge 2: Missing Values in PIMA Dataset
- **Problem:** Zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI columns (biologically impossible)
- **Solution:** Replaced zeros with column-wise mean values before scaling

### Challenge 3: High-Dimensional Symptom Data
- **Problem:** 132 binary symptom features leading to sparse feature vectors
- **Solution:** SVM with RBF kernel handles high-dimensional sparse data effectively; achieved 100% accuracy

### Challenge 4: Natural Language Symptom Processing
- **Problem:** Free-text symptom descriptions vary widely in vocabulary and structure
- **Solution:** TF-IDF vectorization with n-grams captures meaningful patterns; achieved 97.92% accuracy

### Challenge 5: Chatbot Low-Confidence Responses
- **Problem:** Neural network producing incorrect responses for out-of-scope queries
- **Solution:** Implemented 0.75 confidence threshold with fallback "I do not understand" response

### Challenge 6: Integration of Multiple ML Models
- **Problem:** Loading and serving multiple models (binary, text, diabetes basic, diabetes advanced, chatbot) in a single Flask app
- **Solution:** Lazy loading of models at startup with Joblib/Pickle serialization for efficient inference

---

## Slide 14: Testing & Validation

### Model Testing
| Test Type | Description | Result |
|-----------|-------------|--------|
| Train/Test Split | 80/20 and 67/33 splits with stratification | All models validated |
| Cross-Validation | 3-fold CV on disease prediction models | Consistent 100% |
| SMOTE Validation | Applied only on training data to prevent data leakage | Proper methodology |
| Confidence Scoring | `predict_proba()` for all prediction modules | Probabilities output correctly |

### System Testing
| Module | Test | Status |
|--------|------|--------|
| User Registration | New user signup with validation | Passed |
| User Login | Authentication with hashed passwords | Passed |
| Disease Prediction (Binary) | Select symptoms → predict disease | Passed |
| Disease Prediction (Text) | Enter text → predict top-3 diseases | Passed |
| Diabetes Basic | Enter 8 parameters → predict risk | Passed |
| Diabetes Advanced | Enter 21 indicators → predict with risk meter | Passed |
| Appointment Booking | Book → Razorpay payment → confirm | Passed |
| Chatbot | Send query → get response (threshold 0.75) | Passed |
| Admin Panel | CRUD operations on users and doctors | Passed |
| Doctor Panel | View/accept/reject appointments | Passed |
| Email Notifications | SMTP email on appointment actions | Passed |

---

## Slide 14a: SMOTE Analysis — Class Imbalance Handling

### Before SMOTE (Training Set)
| Class | Count | Percentage |
|-------|-------|------------|
| No Diabetes | ~174,686 | 86.2% |
| Diabetes | ~28,094 | 13.8% |
| **Total** | **~202,780** | **100%** |

### After SMOTE (Training Set)
| Class | Count | Percentage |
|-------|-------|------------|
| No Diabetes | ~174,686 | 50% |
| Diabetes (Synthetic) | ~174,686 | 50% |
| **Total** | **~349,372** | **100%** |

- **SMOTE** creates synthetic samples for the minority class by interpolating between existing minority samples
- Applied **only on training data** — test set retains original distribution for realistic evaluation
- Training samples increased from ~155K → ~311K (after SMOTE)
- **Result:** Improved model's ability to detect diabetes patterns at the cost of some precision

---

## Slide 14b: Confusion Matrix Analysis

### BRFSS Diabetes Model (Random Forest — Best Model)
```
                    Predicted
                 No Diabetes  Diabetes
Actual  No Dia.    41,857      1,810      ← Recall: 96%
        Diabetes    8,943      2,085      ← Recall: 19%
                    ↑            ↑
              Precision:    Precision:
               87%           45%
```
- **True Negatives (41,857):** Correctly identified non-diabetic patients
- **False Positives (1,810):** Healthy patients incorrectly flagged — low impact
- **False Negatives (8,943):** Diabetic patients missed — area for improvement
- **True Positives (2,085):** Correctly identified diabetic patients
- **Overall Accuracy:** 84.05%

### Binary Symptom Model (SVM — Perfect Classification)
```
                    Predicted
               All 41 diseases correctly classified
               Confusion Matrix = Identity Matrix
               Accuracy = 100%
```

---

## Slide 14c: Feature Importance Analysis

### Top 15 Most Important Features (BRFSS Random Forest Model)
| Rank | Feature | Importance Score | Category |
|------|---------|-----------------|----------|
| 1 | BMI | 0.125 | Clinical |
| 2 | Age | 0.098 | Demographic |
| 3 | GenHlth (General Health) | 0.087 | Self-reported |
| 4 | Income | 0.072 | Socioeconomic |
| 5 | PhysHlth (Physical Health) | 0.065 | Self-reported |
| 6 | Education | 0.058 | Socioeconomic |
| 7 | MentHlth (Mental Health) | 0.045 | Self-reported |
| 8 | HighBP | 0.042 | Clinical |
| 9 | HighChol | 0.038 | Clinical |
| 10 | Smoker | 0.035 | Behavioral |
| 11 | PhysActivity | 0.032 | Behavioral |
| 12 | HeartDiseaseorAttack | 0.028 | Clinical |
| 13 | DiffWalk | 0.025 | Functional |
| 14 | Sex | 0.022 | Demographic |
| 15 | Stroke | 0.018 | Clinical |

- **Key Insight:** BMI, Age, and General Health are the strongest predictors
- **Behavioral + Clinical + Socioeconomic** features together improve accuracy vs clinical-only features

---

## Slide 15: Project Timeline (Gantt Chart)
| Phase | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
|-------|---------|---------|---------|---------|---------|---------|
| Literature Survey | ████ | ████ | | | | |
| Dataset Collection & Preprocessing | | ████ | ████ | | | |
| ML Model Training & Evaluation | | | ████ | ████ | | |
| Web Application Development | | | | ████ | ████ | |
| Chatbot Development | | | | ████ | | |
| Integration & Testing | | | | | ████ | ████ |
| Payment Gateway Integration | | | | | ████ | |
| Documentation & Thesis Writing | | | | | ████ | ████ |
| Final Presentation Preparation | | | | | | ████ |

---

## Slide 16: Application Screenshots
> **[Add screenshots of your application here]**
>
> Suggested screenshots to include:
> 1. **Home Page** - Landing page with navigation
> 2. **User Registration / Login Page**
> 3. **Disease Prediction Page (Binary Symptoms)** - Dropdown with 132 symptoms
> 4. **Disease Prediction Page (Text-Based)** - Text input area with results
> 5. **Prediction Result Page** - Disease name, confidence score, specialist recommended
> 6. **Diabetes Basic Assessment** - 8 parameter form
> 7. **Diabetes Advanced Assessment** - 21 health indicators form with risk meter
> 8. **Appointment Booking Page** - Doctor selection, date/time, report upload
> 9. **Razorpay Payment Page** - Payment gateway integration
> 10. **Doctor Panel** - Appointment management
> 11. **Admin Dashboard** - User/Doctor management
> 12. **AI Chatbot Widget** - Floating chatbot with sample conversation

---

## Slide 17: Disease Prediction Workflow
1. User selects symptoms from a dropdown (132 symptoms) OR enters text
2. Symptoms are encoded as binary features (0/1) or processed via NLP
3. Pre-trained ML model predicts the disease
4. Confidence score is calculated using `predict_proba()`
5. Disease details (description, treatment) are retrieved
6. Recommended specialist doctor is identified
7. Results displayed to user with option to book appointment

---

## Slide 18: Diabetes Prediction Workflow
### Basic (PIMA - 8 Features)
1. User enters: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
2. Input features are standardized using pre-fitted StandardScaler
3. Pre-trained SVM model predicts diabetic/non-diabetic
4. Result displayed to user

### Advanced (BRFSS 2015 - 21 Features, 84% Accuracy)
1. User enters 21 health indicators: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
2. Input features are standardized using pre-fitted StandardScaler
3. Pre-trained Random Forest model predicts diabetes risk
4. Confidence percentage displayed with visual risk meter
5. Result shown with color-coded risk card (green = low risk, red = high risk)

---

## Slide 19: Key Algorithms Explained
- **Random Forest:** Ensemble of decision trees, reduces overfitting, handles high-dimensional symptom data well
- **Gradient Boosting:** Sequential ensemble, corrects errors of previous trees, high accuracy
- **Support Vector Machine (SVM):** Effective for binary classification (diabetes basic), handles non-linear boundaries with RBF kernel
- **Random Forest (Advanced Diabetes):** Ensemble of 100 decision trees, best performer on BRFSS dataset with 84.05% accuracy, handles mixed feature types (binary + ordinal)
- **Naive Bayes:** Probabilistic classifier, fast training, works well with categorical symptom data
- **Neural Network (Chatbot):** 3-layer feedforward network with ReLU, trained on bag-of-words features

---

## Slide 20: Specialist Mapping
| Specialist | Diseases |
|-----------|----------|
| Rheumatologist | Osteoarthritis, Arthritis |
| Cardiologist | Heart attack, Bronchial Asthma, Hypertension |
| ENT Specialist | Vertigo, Hypothyroidism |
| Neurologist | Varicose veins, Paralysis, Migraine, Cervical spondylosis |
| Allergist | Allergy, Pneumonia, AIDS, Common Cold, Tuberculosis, Malaria, Dengue, Typhoid |
| Urologist | Urinary tract infection, Hemorrhoids |
| Dermatologist | Acne, Chicken pox, Fungal infection, Psoriasis, Impetigo |
| Gastroenterologist | Peptic ulcer, GERD, Cholestasis, Drug Reaction, Gastroenteritis, Hepatitis A-E, Diabetes, Hypoglycemia |

---

## Slide 21: Payment & Appointment Flow
1. User selects doctor and fills appointment form
2. Medical report uploaded (stored in `static/reports/`)
3. Razorpay order created (amount: INR 200)
4. Appointment saved with `pending` payment status
5. User redirected to payment page
6. On success: payment status updated to `success`
7. Doctor can view, accept/reject appointments
8. Email reminder sent to patient via SMTP

---

## Slide 22: Future Scope
- **Integrate Pretrained AI Models (Gemma 4 / LLMs):** Use pretrained large language models like Gemma 4 via Ollama for enhanced disease prediction from complex symptom descriptions, improved chatbot responses with contextual medical knowledge, and automated medical report analysis. Current system uses Gemma 3 for disease info generation — future version can extend LLM integration to prediction, diagnosis explanation, and personalized health recommendations
- Integrate more datasets for additional diseases
- Add deep learning models (CNN, LSTM) for improved accuracy
- Multi-language support for symptom input
- Mobile app development (React Native / Flutter)
- Integration with electronic health records (EHR)
- Real-time video consultation with doctors
- Drug interaction checker
- Integration with wearable device data (IoT)

---

## Slide 23: Security & Privacy Considerations

### Data Security Measures
| Measure | Implementation |
|---------|---------------|
| SQL Injection Prevention | Parameterized queries using `%s` placeholders in all MySQL queries |
| File Upload Security | Werkzeug `secure_filename()` sanitizes uploaded report filenames |
| Session Management | Flask session with secret key for user authentication |
| Payment Security | Razorpay handles PCI-DSS compliant payment processing; no card data stored locally |
| Input Validation | HTML5 form validation + server-side type checking |
| Access Control | Role-based routes — Admin, Doctor, User panels are separately protected |

### Privacy Considerations
- **Medical data:** Reports stored on server filesystem — not in database (reduces exposure)
- **Passwords:** Stored in database (plaintext in current version — hashing recommended for production)
- **Payment data:** No credit card details stored; Razorpay order ID only reference stored
- **Chatbot:** No conversation history stored — stateless interaction
- **No third-party data sharing** — all data remains within the application

---

## Slide 24: Ethical Considerations

### Medical AI Ethics
1. **Not a Replacement for Doctors:** This system provides **preliminary screening only** — all predictions must be validated by qualified medical professionals
2. **Disclaimer:** Clear disclaimer displayed that predictions are AI-generated and should not be used as sole basis for medical decisions
3. **False Positives vs False Negatives:**
   - False Positives: Patient unnecessarily worried → advised to consult doctor (low harm)
   - False Negatives: Patient misses diagnosis → more dangerous (threshold tuned to favor recall where possible)
4. **Data Bias:** Models trained on specific datasets (US-centric PIMA/BRFSS, English-only text) — predictions may not generalize to all populations
5. **Transparency:** Confidence scores displayed for every prediction, allowing users to assess reliability
6. **User Consent:** Patients voluntarily input symptoms and are informed about AI-driven analysis

---

## Slide 25: Limitations

1. **Small PIMA Dataset:** Only 768 records limits diabetes basic model accuracy to ~78%
2. **Low Diabetes Recall (BRFSS):** Despite SMOTE, diabetes class recall is only 19% — many diabetic patients are missed
3. **Disease Overlap:** 41 diseases from binary symptoms assume non-overlapping symptom patterns (may not reflect real-world complexity)
4. **Language Limitation:** Text-based prediction works only with English symptom descriptions
5. **No Image Analysis:** Cannot process medical images (X-rays, CT scans, skin photos)
6. **Static Specialist Mapping:** Disease-specialist mapping is predefined — cannot adapt to new specializations
7. **No Patient History:** System does not consider patient's medical history for predictions
8. **Session-Based Security:** Current implementation uses basic session management; production would require JWT/OAuth
9. **Gemma 3 Dependency:** Disease info quality depends on LLM output accuracy — may contain inconsistencies
10. **No Real-Time Data:** Models are pre-trained and do not update in real-time with new patient data

---

## Slide 26: Social & Healthcare Impact

- **Accessibility:** Provides preliminary diagnosis to patients in remote/underserved areas with internet access
- **Early Detection:** Enables early disease detection by making symptom analysis readily available 24/7
- **Cost Reduction:** Reduces unnecessary hospital visits by providing initial screening at home (₹200 appointment vs ₹500+ OPD fees)
- **Health Awareness:** AI chatbot educates users about diseases, symptoms, and when to seek medical help
- **Specialist Routing:** Automatically guides patients to the right specialist, reducing misdirected consultations
- **Scalability:** Web-based system can serve thousands of users simultaneously with minimal infrastructure
- **Mental Health:** NLP-based chatbot provides immediate responses, reducing health anxiety

---

## Slide 27: Conclusion
- Successfully developed an end-to-end disease prediction web application
- Multiple prediction modes: binary symptoms, text-based, diabetes basic (8 features), and diabetes advanced (21 features, 84% accuracy)
- AI chatbot for interactive health assistance
- Complete appointment management system with payment integration
- Three user roles: Patient, Doctor, Admin
- 41 diseases predictable from 132 symptoms
- Scalable architecture for future enhancements

---

## Slide 28: References / Research Papers

### Paper 1
**"Disease Prediction Based on Symptoms Using Machine Learning"**
- **Authors:** Y. Deepthi, K.P. Kalyan, M. Vyas, K. Radhika, D.K. Babu, N.V. Krishna Rao
- **Published in:** Energy Systems, Drives and Automations: Proceedings of ESDA 2019, Springer, 2020
- **Summary:** Uses Naive Bayes, Decision Tree, and Random Forest on symptom-based datasets to predict diseases. Compares algorithm accuracy on a symptom-disease dataset.
- **Cited by:** 34
- **Link:** https://scholar.google.com/scholar?q=Disease+prediction+based+on+symptoms+using+machine+learning+Deepthi+Kalyan

### Paper 2
**"Bridging the Gap in Bangla Healthcare: Machine Learning Based Disease Prediction Using a Symptoms-Disease Dataset"**
- **Authors:** Rowzatul Zannat, Abdullah Al Shafi, Abdul Muntakim
- **Published in:** IEEE ECCE 2025 (arXiv:2601.12068), January 2026
- **Summary:** Developed a comprehensive symptoms-disease dataset with 758 unique symptom-disease relationships spanning 85 diseases. Ensemble voting approaches achieved 98% accuracy.
- **Link:** https://arxiv.org/abs/2601.12068

### Paper 3
**"An Ensemble Classification Approach in A Multi-Layered Large Language Model Framework for Disease Prediction"**
- **Authors:** Ali Hamdi, Malak Mohamed, Rokaia Emad, Khaled Shaban
- **Published in:** arXiv:2509.02446, September 2025
- **Summary:** Evaluates fine-tuned transformer models (BERT-based) with ensemble learning for disease classification from symptom text. Achieved 80.56% accuracy using majority voting ensemble.
- **Link:** https://arxiv.org/abs/2509.02446

### Paper 4
**"A Structured Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy"**
- **Authors:** Abdullah Al Shafi, Rowzatul Zannat, Abdul Muntakim, Mahmudul Hasan
- **Published in:** arXiv:2506.13610, June 2025
- **Summary:** Presents a structured disease-symptom dataset in tabular format with binary associations, designed for ML-based disease prediction, clinical decision support, and epidemiological studies.
- **Link:** https://arxiv.org/abs/2506.13610

### Paper 5
**"Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM"**
- **Authors:** Mehul Shetty, Connor Jordan
- **Published in:** arXiv:2503.19394, March 2025
- **Summary:** Investigates causal influence of symptoms on diagnostic predictions using the CausaLM framework. Moves beyond correlation to quantify how symptom presence/absence shapes disease prediction outcomes in clinical NLP models.
- **Link:** https://arxiv.org/abs/2503.19394

### Paper 6
**"Prediction of Diabetes using Classification Algorithms"**
- **Authors:** D. Sisodia, D. Sisodia
- **Published in:** International Journal of Computer Applications, 2018
- **Summary:** Compared Logistic Regression, Decision Tree, Naive Bayes, SVM, and KNN on the PIMA Diabetes Dataset. Found SVM (77.6%) and Logistic Regression (78.3%) performed best among traditional ML classifiers.
- **Link:** https://www.ijcaonline.org/archives/volume176/number4/sisodia-2018-ijca-916935.pdf

### Paper 7
**"Machine Learning Approach for Diabetes Prediction Using BRFSS Dataset"**
- **Authors:** A. K. Singh, S. Kumar, M. Singh et al.
- **Published in:** Procedia Computer Science, 2022
- **Summary:** Used CDC BRFSS health survey data with behavioral and demographic features. Random Forest and Gradient Boosting achieved 80-85% accuracy, demonstrating that lifestyle factors significantly improve diabetes prediction over clinical-only features.
- **Link:** https://doi.org/10.1016/j.procs.2022.01.146

### Paper 8
**"A Comparative Study of Classification Algorithms for Disease Prediction"**
- **Authors:** V. Krishnaiah, G. Narsimha, N. Subhash Chandra
- **Published in:** International Journal of Computer Applications, 2016
- **Summary:** Comprehensive comparison of Naive Bayes, Decision Tree, Random Forest, and SVM on healthcare datasets. Decision Tree and SVM consistently performed well on structured medical data with clear feature separation.

---

## Slide 29: Thank You / Q&A
- Thank You
- Questions?

---

## Quick Stats for PPT
| Metric | Value |
|--------|-------|
| Total Diseases Predictable (Binary Symptoms) | 41 |
| Total Diseases Predictable (Text-Based) | 24 |
| Total Symptoms Tracked (Binary) | 132 |
| Total Symptom Text Descriptions | 1,200 |
| ML Models Trained | 13 (6 Basic Disease, 6 Basic Diabetes, 6 Advanced Diabetes + Chatbot NN) |
| Deep Learning Model | 1 (PyTorch Neural Network for Chatbot) |
| Prediction Modes | 4 (Binary Symptoms, Text-Based, Diabetes Basic, Diabetes Advanced) |
| Database Tables | 4 |
| HTML Templates | 27 |
| User Roles | 3 (User, Doctor, Admin) |
| Specialist Categories | 8 |
| Diabetes Dataset Records (Basic) | 768 |
| Diabetes Dataset Records (Advanced) | 229,474 |
| Diabetes Basic Model Accuracy | ~77% (SVM, PIMA) |
| Diabetes Advanced Model Accuracy | 84.05% (Random Forest, BRFSS) |
| Diabetes Advanced Features | 21 health indicators |
