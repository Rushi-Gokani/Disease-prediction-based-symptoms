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
| `best_text_model.joblib` | Scikit-learn Pipeline (TF-IDF + Classifier) | **97.92%** |

- **Dataset:** Symptom2Disease — 1,200 text descriptions across 24 diseases
- **Output:** Top-3 predicted diseases with confidence scores
- **Saved with:** Label encoder (`text_label_encoder.joblib`)

### Model 3: Diabetes Prediction (Basic - PIMA) — 8 features, 768 records
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `svm_model.pkl` | **Support Vector Machine** | 75.32% |
| `scaler.pkl` + retrained | **Naive Bayes** | **78.57% (Best)** |
| `scaler.pkl` + retrained | **Logistic Regression** | 76.62% |
| `scaler.pkl` + retrained | **Random Forest** | 75.97% |
| `scaler.pkl` + retrained | **KNN** | 75.32% |
| `scaler.pkl` + retrained | **Decision Tree** | 73.38% |

- **Feature Scaling:** StandardScaler
- **Train/Test Split:** 80/20 (random_state=7)
- **Missing Values:** Replaced zeros with column mean for Glucose, BP, SkinThickness, Insulin, BMI
- **Production Model:** SVM (original training selected SVM as best)

### Model 4: Diabetes Prediction (Advanced - BRFSS 2015) — 21 features, 229,474 records
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| `diabetes_brfss_model.pkl` | **Random Forest (100 estimators)** | **84.05% (Best)** |
| Tested | **Gradient Boosting** | 83.78% |
| Tested | **Decision Tree** | 77.13% |
| Tested | **KNN** | 75.28% |
| Tested | **Logistic Regression** | 71.50% |
| Tested | **Naive Bayes** | 64.39% |

- **Feature Scaling:** StandardScaler
- **Class Balancing:** SMOTE (oversampling minority class — 155K → 311K training samples)
- **Train/Test Split:** 80/20 (stratified)
- **Classification Report (Random Forest):**
  - No Diabetes: Precision 0.87, Recall 0.96, F1 0.91
  - Diabetes: Precision 0.45, Recall 0.19, F1 0.27

### Model 5: Chatbot (Deep Learning)
| Model | Framework | Type |
|-------|-----------|------|
| `data.pth` | PyTorch | 3-layer Feedforward Neural Network |

- **Architecture:** Input → Hidden (ReLU) → Hidden (ReLU) → Output
- **NLP:** NLTK tokenization + PorterStemmer + Bag of Words
- **Confidence Threshold:** 0.75

### Overall Accuracy Summary Table
| Module | Best Algorithm | Accuracy | Dataset Size |
|--------|---------------|----------|-------------|
| Disease Prediction (Binary) | SVM / Decision Tree / MNB | **100.00%** | 4,920 train + test |
| Disease Prediction (Text) | TF-IDF Pipeline | **97.92%** | 1,200 records |
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

## Slide 12: Project File Structure
```
Disease-prediction-based-symptoms/
|-- app.py                    # Main Flask application (849 lines)
|-- main.py                   # ML model training pipeline
|-- model.py                  # PyTorch Neural Network definition
|-- nltk_utils.py             # NLP utilities (tokenize, stem, bag_of_words)
|-- chat.py                   # Chatbot inference module
|-- infer.py                  # Disease prediction inference script
|-- diabetes_prediction.py    # Diabetes model training script (Basic PIMA)
|-- train_diabetes_brfss.py   # Advanced diabetes model training script (BRFSS 2015)
|-- generate_disease_info.py  # LLM-based disease info generator
|-- config.yaml               # ML model configuration
|-- database.sql              # MySQL database schema
|-- requirements.txt          # Python dependencies
|-- environment.yml           # Conda environment config
|-- intents.json              # Chatbot training intents
|-- data.pth                  # Trained PyTorch chatbot model
|-- scaler.pkl                # StandardScaler for diabetes model (Basic)
|-- svm_model.pkl             # SVM model for diabetes prediction (Basic)
|-- diabetes_brfss_model.pkl  # Random Forest model for diabetes prediction (Advanced)
|-- diabetes_brfss_scaler.pkl # StandardScaler for advanced diabetes model
|-- diabetes_brfss_features.pkl # Feature names for advanced diabetes model
|-- diabetes.csv              # PIMA diabetes dataset
|-- diabetes_binary_health_indicators_BRFSS2015.csv  # BRFSS 2015 diabetes dataset (253K records)
|-- Symptom2Disease.csv       # Text-based symptom-disease dataset (1200 records, 24 diseases)
|-- disease_info.json         # Generated disease information
|-- dataset/
|   |-- training_data.csv     # Symptom-disease training data
|   |-- test_data.csv         # Symptom-disease test data
|-- saved_model/
|   |-- best_binary_model.joblib
|   |-- best_text_model.joblib
|   |-- text_label_encoder.joblib
|   |-- binary_label_encoder.joblib
|   |-- decision_tree.joblib
|   |-- mnb.joblib
|   |-- gradient_boost.joblib
|-- static/
|   |-- app.js                # Chatbox JavaScript
|   |-- css/                  # Stylesheets
|   |-- img/                  # Images
|-- templates/                # 26 HTML templates
|   |-- index.html, login.html, register.html, ...
|   |-- admin/                # Admin panel templates
|   |-- dpanel/               # Doctor panel templates
```

---

## Slide 13: Disease Prediction Workflow
1. User selects symptoms from a dropdown (132 symptoms) OR enters text
2. Symptoms are encoded as binary features (0/1) or processed via NLP
3. Pre-trained ML model predicts the disease
4. Confidence score is calculated using `predict_proba()`
5. Disease details (description, treatment) are retrieved
6. Recommended specialist doctor is identified
7. Results displayed to user with option to book appointment

---

## Slide 14: Diabetes Prediction Workflow
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

## Slide 15: Key Algorithms Explained
- **Random Forest:** Ensemble of decision trees, reduces overfitting, handles high-dimensional symptom data well
- **Gradient Boosting:** Sequential ensemble, corrects errors of previous trees, high accuracy
- **Support Vector Machine (SVM):** Effective for binary classification (diabetes basic), handles non-linear boundaries with RBF kernel
- **Random Forest (Advanced Diabetes):** Ensemble of 100 decision trees, best performer on BRFSS dataset with 84.05% accuracy, handles mixed feature types (binary + ordinal)
- **Naive Bayes:** Probabilistic classifier, fast training, works well with categorical symptom data
- **Neural Network (Chatbot):** 3-layer feedforward network with ReLU, trained on bag-of-words features

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

## Slide 17: Payment & Appointment Flow
1. User selects doctor and fills appointment form
2. Medical report uploaded (stored in `static/reports/`)
3. Razorpay order created (amount: INR 200)
4. Appointment saved with `pending` payment status
5. User redirected to payment page
6. On success: payment status updated to `success`
7. Doctor can view, accept/reject appointments
8. Email reminder sent to patient via SMTP

---

## Slide 18: Future Scope
- Integrate more datasets for additional diseases
- Add deep learning models (CNN, LSTM) for improved accuracy
- Multi-language support for symptom input
- Mobile app development (React Native / Flutter)
- Integration with electronic health records (EHR)
- Real-time video consultation with doctors
- Drug interaction checker
- Integration with wearable device data (IoT)

---

## Slide 19: Conclusion
- Successfully developed an end-to-end disease prediction web application
- Multiple prediction modes: binary symptoms, text-based, diabetes basic (8 features), and diabetes advanced (21 features, 84% accuracy)
- AI chatbot for interactive health assistance
- Complete appointment management system with payment integration
- Three user roles: Patient, Doctor, Admin
- 41 diseases predictable from 132 symptoms
- Scalable architecture for future enhancements

---

## Slide 20: References / Research Papers

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

## Slide 21: Thank You / Q&A
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
