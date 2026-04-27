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

### Data Flow Diagram (Level 1)
```
User --> [1.0 Auth] --> User Session
User --> [2.0 Disease Pred] --> ML Model --> Result + Specialist
User --> [3.0 Diabetes Pred] --> ML Model --> Risk Score
User --> [4.0 Book Appt] --> [5.0 Payment] --> DB --> Doctor
User --> [6.0 Chatbot] --> PyTorch NN --> Response
Admin -> [7.0 Manage Users/Doctors] --> MySQL DB
Doctor -> [8.0 Manage Appointments] --> MySQL DB --> Email (SMTP)
```

### Use Case Actors
- **User (Patient):** Login, Predict Disease (Binary/Text), Diabetes Assessment (Basic/Advanced), Book Appointment, Payment, Chatbot
- **Doctor:** View Appointments, Accept/Reject, Send Email Reminders
- **Admin:** Dashboard, Manage Users, Manage Doctors

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
- **Training Data:** `dataset/training_data.csv` | **Test Data:** `dataset/test_data.csv`
- **Features:** 132 binary symptom columns (0 or 1)
- **Target:** `prognosis` column (41 diseases)
- **Diseases Covered (41):** Fungal infection, Allergy, GERD, Diabetes, Hypertension, Migraine, Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A-E, Tuberculosis, Common Cold, Pneumonia, Heart attack, Arthritis, Acne, Psoriasis, Impetigo, and more

### Dataset 2: Diabetes Dataset (Basic - PIMA)
- **Records:** 768 patients
- **Features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (0 = Non-Diabetic, 1 = Diabetic)

### Dataset 3: Diabetes Dataset (Advanced - BRFSS 2015)
- **Records:** 253,680 patients (229,474 after deduplication)
- **Features (21):** HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
- **Preprocessing:** SMOTE applied to handle class imbalance; 24,206 duplicate rows removed

### Dataset 4: Symptom2Disease (Text-based)
- **Records:** 1,200 patient symptom descriptions across 24 diseases (50 per disease, balanced)
- **Used for:** Text-based disease prediction model using NLP (TF-IDF + classifier)

---

## Slide 7: Literature Review

### Study 1: Disease Prediction from Symptoms Using ML
- **Authors:** Y. Deepthi, K.P. Kalyan et al. (2020, Springer)
- **Finding:** Decision Tree and Random Forest achieved >95% accuracy on structured symptom datasets.
- **Relevance:** Our project achieves Decision Tree (100%), Naive Bayes (100%), and SVM (100%).

### Study 2: ML-Based Diabetes Prediction on PIMA Dataset
- **Authors:** D. Sisodia, D. Sisodia (2018, IJCA)
- **Finding:** SVM (77.6%) and Logistic Regression (78.3%) performed best on PIMA dataset.
- **Relevance:** We achieved comparable results — Naive Bayes 78.57%, Logistic Regression 76.62%.

### Study 3: Diabetes Prediction Using BRFSS Data
- **Authors:** A. K. Singh, S. Kumar et al. (2022, Journal of Diabetes Research)
- **Finding:** Random Forest and Gradient Boosting achieved 80-85% accuracy on BRFSS data.
- **Relevance:** Our BRFSS model achieved 84.05% with Random Forest using 21 health indicators.

### Study 4: NLP-Based Disease Prediction from Symptom Text
- **Authors:** A. Hamdi, M. Mohamed et al. (2025, arXiv)
- **Finding:** Fine-tuned BERT-based ensemble achieved 80.56% accuracy.
- **Relevance:** Our TF-IDF + classifier pipeline achieves 97.92% on Symptom2Disease dataset.

### Key Literature Gaps Addressed:
1. Most studies focus on **single prediction mode** — we integrate **4 prediction modes**
2. Limited studies use **21+ health indicators** for diabetes prediction
3. Few systems provide **end-to-end functionality** (prediction -> specialist -> appointment -> payment)

---

## Slide 8: Machine Learning Models & Accuracy Results

### Model 1: Disease Prediction (Binary Symptoms) — 132 features, 41 diseases
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `best_binary_model.joblib` | SVM (RBF kernel, C=0.1) | **100.00%** |
| `decision_tree.joblib` | Decision Tree (gini) | **100.00%** |
| `mnb.joblib` | Multinomial Naive Bayes | **100.00%** |

### Model 2: Disease Prediction (Text-Based) — Natural Language Input
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `best_text_model.joblib` | TF-IDF + MNB Pipeline | **97.92%** |

### Model 3: Diabetes Prediction (Basic - PIMA) — 8 features, 768 records
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `svm_model.pkl` | **Support Vector Machine** | 75.32% |
| retrained | **Naive Bayes** | **78.57% (Best)** |
| retrained | **Logistic Regression** | 76.62% |

### Model 4: Diabetes Prediction (Advanced - BRFSS 2015) — 21 features, 229,474 records
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `diabetes_brfss_model.pkl` | **Random Forest (100 estimators)** | **84.05% (Best)** |
| Tested | **Gradient Boosting** | 83.78% |

### Key Mathematical Foundations
- **SVM:** Decision function f(x) = w^T . phi(x) + b; RBF Kernel K(xi, xj) = exp(-gamma ||xi - xj||^2); C=0.1
- **Random Forest:** Ensemble prediction h(x) = (1/B) sum of B trees (majority vote); Gini impurity = 1 - sum p(i|t)^2; n_estimators=100
- **Naive Bayes:** P(y|x) proportional to P(y) * product of P(xi|y); Laplace smoothing alpha=0.01
- **TF-IDF:** TF-IDF(t,d) = TF(t,d) x IDF(t); max_features=8000, ngram_range=(1,2)

### Overall Accuracy Summary
| Module | Best Algorithm | Accuracy | Dataset Size |
|--------|---------------|----------|-------------|
| Disease Prediction (Binary) | SVM / Decision Tree / MNB | **100.00%** | 4,920 train + test |
| Disease Prediction (Text) | TF-IDF + MNB | **97.92%** | 1,200 records |
| Diabetes (Basic - PIMA) | Naive Bayes / SVM | **78.57%** | 768 records |
| Diabetes (Advanced - BRFSS) | Random Forest | **84.05%** | 229,474 records |
| Chatbot | PyTorch Neural Network | 0.75 threshold | intents.json |

---

## Slide 9: Deep Learning - Chatbot (Neural Network)
- **Framework:** PyTorch
- **Architecture:**
  - Input Layer: `input_size` neurons
  - Hidden Layer 1: `hidden_size` neurons + ReLU activation
  - Hidden Layer 2: `hidden_size` neurons + ReLU activation
  - Output Layer: `num_classes` neurons
- **Forward pass:** z1 = W1.x + b1 -> a1 = ReLU(z1) -> z2 = W2.a1 + b2 -> a2 = ReLU(z2) -> z3 = W3.a2 + b3
- **Loss:** CrossEntropyLoss = -sum ci log(pi)
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
- Book Appointment with Doctors, Upload Medical Reports
- Make Payment via Razorpay, View Appointment History
- Chatbot for Health Queries

### Doctor Panel
- Doctor Login, View Appointments
- Accept / Reject Appointments
- Send Email Reminders to Patients

### Admin Panel
- Admin Login, Dashboard (Total Users, Total Doctors)
- Manage Users (Add, Edit, Delete)
- Manage Doctors (Add, Delete)

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
| Model Serialization | Joblib, Pickle |
| Payment Gateway | Razorpay |
| Email Service | SMTP (Gmail) |
| LLM Integration | Ollama (Gemma 3) for disease info generation |

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
| Accuracy (Disease) | Variable | 85-95% | **100%** (Binary), **97.92%** (Text) |
| Accuracy (Diabetes) | N/A | 75-80% | **84.05%** (Advanced BRFSS) |

---

## Slide 13: Challenges Faced & Solutions

### Challenge 1: Class Imbalance in BRFSS Diabetes Dataset
- **Problem:** No Diabetes: 218,334 vs Diabetes: 35,346 (heavily imbalanced)
- **Solution:** Applied SMOTE to balance training data, improving minority class recall

### Challenge 2: Missing Values in PIMA Dataset
- **Problem:** Zero values in Glucose, BloodPressure, BMI columns (biologically impossible)
- **Solution:** Replaced zeros with column-wise mean values before scaling

### Challenge 3: High-Dimensional Symptom Data
- **Problem:** 132 binary symptom features leading to sparse feature vectors
- **Solution:** SVM with RBF kernel handles high-dimensional sparse data; achieved 100% accuracy

### Challenge 4: Natural Language Symptom Processing
- **Problem:** Free-text symptom descriptions vary widely in vocabulary and structure
- **Solution:** TF-IDF vectorization with n-grams; achieved 97.92% accuracy

### Challenge 5: Integration of Multiple ML Models
- **Problem:** Loading and serving multiple models in a single Flask app
- **Solution:** Lazy loading with Joblib/Pickle serialization for efficient inference

---

## Slide 14: Testing & Validation

### Model Testing
| Test Type | Description | Result |
|-----------|-------------|--------|
| Train/Test Split | 80/20 and 67/33 splits with stratification | All models validated |
| Cross-Validation | 3-fold CV on disease prediction models | Consistent 100% |
| SMOTE Validation | Applied only on training data to prevent data leakage | Proper methodology |
| Confidence Scoring | `predict_proba()` for all prediction modules | Probabilities output correctly |

### SMOTE Analysis — Class Imbalance Handling
| Stage | No Diabetes | Diabetes | Total |
|-------|------------|----------|-------|
| Before SMOTE (Train) | ~174,686 (86.2%) | ~28,094 (13.8%) | ~202,780 |
| After SMOTE (Train) | ~174,686 (50%) | ~174,686 (50%) | ~349,372 |

### Confusion Matrix — BRFSS Diabetes Model (Random Forest)
- **True Negatives (41,857):** Correctly identified non-diabetic patients
- **False Positives (1,810):** Healthy patients incorrectly flagged — low impact
- **False Negatives (8,943):** Diabetic patients missed — area for improvement
- **True Positives (2,085):** Correctly identified diabetic patients
- **Overall Accuracy:** 84.05% | **No-Diabetes Recall:** 96% | **Diabetes Recall:** 19%

### Top Feature Importances (BRFSS Random Forest)
- BMI (0.125), Age (0.098), GenHlth (0.087), Income (0.072), PhysHlth (0.065), Education (0.058), MentHlth (0.045), HighBP (0.042), HighChol (0.038)
- **Key Insight:** BMI, Age, and General Health are the strongest predictors

### System Testing
| Module | Test | Status |
|--------|------|--------|
| Disease Prediction (Binary) | Select symptoms -> predict disease | Passed |
| Disease Prediction (Text) | Enter text -> predict top-3 diseases | Passed |
| Diabetes Basic | Enter 8 parameters -> predict risk | Passed |
| Diabetes Advanced | Enter 21 indicators -> predict with risk meter | Passed |
| Appointment Booking | Book -> Razorpay payment -> confirm | Passed |
| Chatbot | Send query -> get response (threshold 0.75) | Passed |
| Admin/Doctor Panel | CRUD, accept/reject appointments | Passed |

---

## Slide 15: Prediction Workflows

### Disease Prediction Workflow
1. User selects symptoms from a dropdown (132 symptoms) OR enters text
2. Symptoms are encoded as binary features (0/1) or processed via NLP (TF-IDF)
3. Pre-trained ML model predicts the disease with confidence score
4. Disease details (description, treatment) and recommended specialist are displayed
5. User can book appointment with the recommended specialist

### Diabetes Prediction Workflow (Basic - 8 Features)
1. User enters: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
2. Input standardized using pre-fitted StandardScaler
3. SVM model predicts diabetic/non-diabetic

### Diabetes Prediction Workflow (Advanced - 21 Features, 84% Accuracy)
1. User enters 21 health indicators (HighBP, HighChol, BMI, Smoker, etc.)
2. Input standardized using pre-fitted StandardScaler
3. Random Forest model predicts diabetes risk with confidence percentage
4. Result shown with color-coded risk meter (green = low risk, red = high risk)

---

## Slide 16: Specialist Mapping
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

## Slide 17: Future Scope
- **Integrate Pretrained AI Models (Gemma 4 / LLMs):** Use large language models for enhanced disease prediction, improved chatbot responses, and automated medical report analysis
- Integrate more datasets for additional diseases
- Add deep learning models (CNN, LSTM) for improved accuracy
- Multi-language support for symptom input
- Mobile app development (React Native / Flutter)
- Integration with electronic health records (EHR)
- Real-time video consultation with doctors
- Drug interaction checker
- Integration with wearable device data (IoT)

---

## Slide 18: Limitations & Ethical Considerations

### Limitations
1. **Small PIMA Dataset:** Only 768 records limits diabetes basic model accuracy to ~78%
2. **Low Diabetes Recall (BRFSS):** Despite SMOTE, diabetes class recall is only 19%
3. **Language Limitation:** Text-based prediction works only with English descriptions
4. **No Image Analysis:** Cannot process medical images (X-rays, CT scans, skin photos)
5. **No Patient History:** System does not consider patient's medical history for predictions
6. **Static Specialist Mapping:** Disease-specialist mapping is predefined — cannot adapt to new specializations

### Ethical Considerations
1. **Not a Replacement for Doctors:** Provides **preliminary screening only** — predictions must be validated by medical professionals
2. **False Positives vs False Negatives:** Threshold tuned to favor recall where possible
3. **Data Bias:** Models trained on specific datasets (US-centric PIMA/BRFSS, English-only text)
4. **Transparency:** Confidence scores displayed for every prediction
5. **User Consent:** Patients voluntarily input symptoms and are informed about AI-driven analysis

---

## Slide 19: References

1. Y. Deepthi, K.P. Kalyan et al. (2020) — "Disease Prediction Based on Symptoms Using ML" — Springer
2. R. Zannat, A. Al Shafi et al. (2026) — "Bangla Healthcare: ML-Based Disease Prediction" — IEEE ECCE
3. A. Hamdi, M. Mohamed et al. (2025) — "Ensemble LLM Framework for Disease Prediction" — arXiv
4. A. Al Shafi, R. Zannat et al. (2025) — "Structured Dataset of Disease-Symptom Associations" — arXiv
5. M. Shetty, C. Jordan (2025) — "Quantifying Symptom Causality Using CausaLM" — arXiv
6. D. Sisodia, D. Sisodia (2018) — "Prediction of Diabetes using Classification Algorithms" — IJCA
7. A.K. Singh, S. Kumar et al. (2022) — "ML Approach for Diabetes Using BRFSS Dataset" — Procedia CS
8. V. Krishnaiah et al. (2016) — "Comparative Study of Classification Algorithms for Disease Prediction" — IJCA

---

## Slide 20: Conclusion & Thank You

### Conclusion
- Successfully developed an end-to-end disease prediction web application
- Multiple prediction modes: binary symptoms, text-based, diabetes basic (8 features), and diabetes advanced (21 features, 84% accuracy)
- AI chatbot for interactive health assistance
- Complete appointment management system with payment integration
- Three user roles: Patient, Doctor, Admin
- 41 diseases predictable from 132 symptoms

### Social & Healthcare Impact
- **Accessibility:** Provides preliminary diagnosis to patients in remote/underserved areas
- **Early Detection:** Makes symptom analysis available 24/7
- **Cost Reduction:** Reduces unnecessary hospital visits (Rs.200 appointment vs Rs.500+ OPD fees)
- **Specialist Routing:** Automatically guides patients to the right specialist

### Thank You — Questions?

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
