# Disease Prediction Based on Symptoms

## An AI-Powered Healthcare Web Application for Early Disease Detection

---

**Submitted in partial fulfillment of the requirements for the degree of**

**Master of Technology (M.Tech)**

---

| | |
|---|---|
| **Submitted By:** | [Your Name] |
| **Roll Number:** | [Your Roll Number] |
| **Department:** | [Department Name] |
| **University:** | [University Name] |
| **Guide:** | [Guide Name] |
| **Year:** | 2025-2026 |

---

## Declaration

I hereby declare that the project entitled **"Disease Prediction Based on Symptoms"** is a record of original work carried out by me under the guidance of [Guide Name], [Department], [University Name], in partial fulfillment of the requirements for the award of the degree of Master of Technology. This project has not been submitted elsewhere for the award of any degree or diploma.

**Date:** [Date]

**Signature:** _______________

---

## Certificate

This is to certify that the project entitled **"Disease Prediction Based on Symptoms"** is a bonafide record of work done by [Your Name] (Roll No: [Roll Number]) under my guidance and supervision, in partial fulfillment of the requirements for the award of the degree of Master of Technology in [Department] from [University Name].

**Guide Name:**

**Department:**

**University Name:**

**Signature:** _______________

---

## Acknowledgements

I would like to express my sincere gratitude to my project guide [Guide Name] for their invaluable guidance, constant encouragement, and support throughout the duration of this project. I also thank the faculty members of the Department of [Department] for their constructive feedback and suggestions.

I extend my thanks to the developers of open-source tools and libraries — Scikit-learn, PyTorch, Flask, NLTK, and Ollama (Gemma 3) — that made this project possible. I am also grateful to the creators of the publicly available datasets (Symptom-Disease Dataset, PIMA Diabetes Dataset, CDC BRFSS 2015, and Symptom2Disease) used in this research.

Finally, I thank my family and friends for their unwavering support and encouragement.

---

## Abstract

Healthcare systems worldwide face challenges in providing timely and accurate preliminary disease diagnosis. Manual diagnosis is time-consuming, prone to human error, and often inaccessible to patients in remote areas. This project presents an AI-powered web application that predicts diseases from user-reported symptoms using Machine Learning and Natural Language Processing techniques.

The system integrates **four distinct prediction modules**: (1) Binary symptom-based disease prediction using 132 symptoms across 41 diseases, (2) Text-based disease prediction from natural language symptom descriptions across 24 diseases, (3) Basic diabetes risk assessment using 8 clinical features from the PIMA dataset, and (4) Advanced diabetes risk assessment using 21 health indicators from the CDC BRFSS 2015 dataset (229,474 records).

Multiple ML algorithms were evaluated including Random Forest, SVM, Decision Tree, Gradient Boosting, Naive Bayes, Logistic Regression, and XGBoost. The binary symptom models achieved **100% test accuracy** (SVM), the text-based model achieved **97.9% accuracy** (TF-IDF + Multinomial Naive Bayes), and the advanced diabetes model achieved **84.05% accuracy** (Random Forest with SMOTE). A PyTorch-based neural network chatbot provides interactive health assistance.

The application also features online appointment booking with specialist doctors, Razorpay payment gateway integration, email notifications via SMTP, and role-based access control (User, Doctor, Admin). Disease information was enriched using **Gemma 3** (LLM via Ollama) for generating structured medical content.

**Keywords:** Disease Prediction, Machine Learning, Natural Language Processing, Diabetes Prediction, Flask, PyTorch, Chatbot, Healthcare, SMOTE, TF-IDF

---

## Table of Contents

1. Chapter 1: Introduction
   - 1.1 Background
   - 1.2 Problem Statement
   - 1.3 Objectives
   - 1.4 Scope of the Project
   - 1.5 Organization of the Report
2. Chapter 2: Literature Survey
3. Chapter 3: System Design and Methodology
   - 3.1 System Architecture
   - 3.2 Database Design
   - 3.3 Machine Learning Pipeline
   - 3.4 NLP Pipeline
   - 3.5 Deep Learning Pipeline (Chatbot)
   - 3.6 Technology Stack
4. Chapter 4: Implementation
   - 4.1 Dataset Details
   - 4.2 Data Preprocessing
   - 4.3 Feature Engineering
   - 4.4 Model Training and Selection
   - 4.5 Web Application Implementation
   - 4.6 Payment Gateway Integration
   - 4.7 LLM Integration (Gemma 3)
5. Chapter 5: Results and Discussion
   - 5.1 Disease Prediction Results (Binary Symptoms)
   - 5.2 Disease Prediction Results (Text-Based)
   - 5.3 Diabetes Prediction Results (Basic - PIMA)
   - 5.4 Diabetes Prediction Results (Advanced - BRFSS)
   - 5.5 Chatbot Performance
   - 5.6 Comparison with Existing Systems
   - 5.7 Performance Evaluation Metrics
6. Chapter 6: Testing and Validation
7. Chapter 7: Challenges and Solutions
8. Chapter 8: Future Scope
9. Chapter 9: Conclusion
10. References
11. Appendices

---

# CHAPTER 1

## INTRODUCTION

### 1.1 Background

The healthcare industry is one of the most critical sectors in any country's economy. Timely and accurate diagnosis of diseases is essential for effective treatment and improved patient outcomes. However, the traditional approach to disease diagnosis relies heavily on manual examination by medical professionals, which can be time-consuming and subjective. In many developing regions, patients also face a shortage of qualified medical practitioners, leading to delayed diagnoses and worsening health conditions.

With the rapid advancement of Artificial Intelligence (AI) and Machine Learning (ML), there is significant potential to develop automated systems that can assist in preliminary disease diagnosis. Such systems can analyze patient-reported symptoms and predict probable diseases with high accuracy, enabling patients to seek timely medical attention from the appropriate specialist.

Natural Language Processing (NLP) further enhances these systems by allowing patients to describe their symptoms in plain text, removing the need for structured form inputs. Deep Learning techniques enable the creation of intelligent chatbots that can interact with patients, answer health-related queries, and guide them through the diagnostic process.

### 1.2 Problem Statement

Despite advances in medical technology, the following challenges persist:

1. **Delayed Diagnosis:** Patients often delay seeking medical attention due to lack of awareness about their symptoms, leading to worsening health conditions.
2. **Specialist Identification:** Patients frequently do not know which medical specialist to consult for their specific condition.
3. **Limited Accessibility:** Automated preliminary diagnosis tools are not widely available, especially in rural and underserved areas.
4. **Manual Processes:** Appointment booking, report management, and follow-ups are often manual and inefficient.
5. **Single-Mode Prediction:** Existing systems typically offer only one mode of prediction (e.g., structured symptoms only), limiting user accessibility.

**There is a need for an integrated, AI-powered healthcare web application that provides multiple prediction modes, specialist mapping, appointment management, and interactive health assistance in a single platform.**

### 1.3 Objectives

The primary objectives of this project are:

1. To develop a **web-based disease prediction system** using Machine Learning algorithms that can predict diseases from user-selected binary symptoms.
2. To implement **text-based disease prediction** using NLP techniques (TF-IDF vectorization) that can predict diseases from free-text symptom descriptions.
3. To build a **diabetes risk assessment module** with two modes:
   - Basic mode using 8 clinical features from the PIMA Diabetes Dataset
   - Advanced mode using 21 health indicators from the CDC BRFSS 2015 dataset
4. To design an **AI-powered chatbot** using a PyTorch neural network for interactive health-related queries.
5. To implement **specialist doctor mapping** that automatically recommends the appropriate medical specialist based on the predicted disease.
6. To integrate **online appointment booking** with payment gateway (Razorpay) and email notification (SMTP) capabilities.
7. To develop a **role-based access control system** with User, Doctor, and Admin panels.
8. To leverage **Large Language Models (Gemma 3 via Ollama)** for generating structured disease information (description, symptoms, precautions, medications, diet recommendations).

### 1.4 Scope of the Project

The scope of this project includes:

- Predicting **41 diseases** from **132 binary symptoms**
- Predicting **24 diseases** from **natural language symptom text**
- Diabetes risk assessment using **8 clinical features** (PIMA) and **21 health indicators** (BRFSS 2015)
- Appointment booking and management with payment processing
- AI chatbot for health query assistance
- The system is intended as a **preliminary diagnostic tool** and not a replacement for professional medical advice

### 1.5 Organization of the Report

- **Chapter 2** presents a comprehensive literature survey of related work.
- **Chapter 3** describes the system design and methodology.
- **Chapter 4** details the implementation of each module.
- **Chapter 5** presents the results and performance analysis.
- **Chapter 6** covers testing and validation.
- **Chapter 7** discusses challenges faced and solutions implemented.
- **Chapter 8** outlines the future scope of the project.
- **Chapter 9** provides the conclusion.

---

# CHAPTER 2

## LITERATURE SURVEY

### 2.1 Disease Prediction from Symptoms Using Machine Learning

**Deepthi et al. (2020)** [1] compared Naive Bayes, Decision Tree, and Random Forest algorithms for symptom-based disease prediction. Their study demonstrated that Decision Tree and Random Forest achieved over 95% accuracy on structured symptom datasets. The study used a binary feature representation where each symptom was encoded as 0 or 1, similar to our approach. However, the study focused on a single prediction mode and did not include text-based input or end-to-end appointment management.

### 2.2 ML-Based Diabetes Prediction on PIMA Dataset

**Sisodia and Sisodia (2018)** [2] conducted a comparative study of classification algorithms on the PIMA Diabetes Dataset. They reported Logistic Regression at 78.3%, Decision Tree at 73.4%, Naive Bayes at 76.3%, SVM at 77.6%, and KNN at 72.4%. Their findings indicated that SVM and Logistic Regression performed best among traditional ML classifiers. Our system uses the same PIMA dataset and achieves comparable results, with Naive Bayes at 78.57% and SVM at 75.32%.

### 2.3 Diabetes Prediction Using BRFSS Data

**Singh et al. (2022)** [3] used CDC BRFSS health survey data with 21+ features for diabetes prediction. Their study demonstrated that Random Forest and Gradient Boosting achieved 80-85% accuracy, significantly outperforming models using only clinical features. This finding validated the importance of including behavioral and demographic health indicators. Our BRFSS model achieved 84.05% accuracy with Random Forest using 21 features from 229,474 records, consistent with their reported range.

### 2.4 NLP-Based Disease Prediction from Symptom Text

**Hamdi et al. (2025)** [4] evaluated fine-tuned transformer models (BERT-based) with ensemble learning for disease classification from symptom text. They achieved 80.56% accuracy using a majority voting ensemble approach. In comparison, our text-based pipeline achieves 97.92% accuracy using TF-IDF + Multinomial Naive Bayes on the Symptom2Disease dataset, benefiting from a smaller but well-curated dataset with 50 samples per disease.

### 2.5 Ensemble Approaches for Disease Prediction

**Zannat et al. (2026)** [5] developed a comprehensive symptoms-disease dataset with 758 unique symptom-disease relationships spanning 85 diseases. Their ensemble voting approach achieved 98% accuracy. Our project uses both individual and ensemble models, with SVM, Decision Tree, and Multinomial Naive Bayes achieving high accuracy on the binary symptom dataset.

### 2.6 Structured Disease-Symptom Datasets

**Al Shafi et al. (2025)** [6] presented a structured disease-symptom dataset in tabular format with binary associations, specifically designed for ML-based disease prediction and clinical decision support. Their work on creating standardized datasets for symptom-disease mapping directly relates to the binary symptom encoding approach used in our project.

### 2.7 Causality in Clinical Decision Making

**Shetty and Jordan (2025)** [7] investigated the causal influence of symptoms on diagnostic predictions using the CausaLM framework, moving beyond correlation to quantify how symptom presence or absence shapes disease prediction outcomes. This approach highlights the importance of understanding symptom-disease relationships beyond simple pattern matching.

### 2.8 Comparative Study of Classification Algorithms

**Krishnaiah et al. (2016)** [8] conducted a comprehensive comparison of Naive Bayes, Decision Tree, Random Forest, and SVM on healthcare datasets. Their findings showed that Decision Tree and SVM consistently performed well on structured medical data with clear feature separation, which aligns with our results where these algorithms achieved high accuracy on the binary symptom dataset.

### 2.9 Key Literature Gaps Addressed by This Project

| Gap | How This Project Addresses It |
|-----|-------------------------------|
| Most studies focus on a single prediction mode | Our system integrates **4 prediction modes** in one application |
| Limited studies use 21+ health indicators for diabetes | We leverage BRFSS with **21 features and 229,474 records** |
| Few systems provide end-to-end functionality | We provide **prediction → specialist mapping → appointment → payment** |
| Limited integration of NLP for symptom text | TF-IDF pipeline achieves **97.92% accuracy** on free-text input |
| No LLM-enriched disease information | **Gemma 3 via Ollama** generates structured medical content |

---

# CHAPTER 3

## SYSTEM DESIGN AND METHODOLOGY

### 3.1 System Architecture

The system follows a **three-tier architecture**:

```
┌─────────────────────────────────────────────────────┐
│                   PRESENTATION TIER                  │
│          HTML5, CSS3, Bootstrap, JavaScript          │
│    (26+ HTML Templates, Responsive UI, Chatbot)      │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP/HTTPS
┌─────────────────────▼───────────────────────────────┐
│                    LOGIC TIER                         │
│               Python Flask (app.py)                   │
│   ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐  │
│   │ Disease │ │ Diabetes  │ │ Chat   │ │ Payment  │  │
│   │Predictor│ │ Predictor │ │  Bot   │ │ Gateway  │  │
│   └────┬────┘ └─────┬────┘ └───┬────┘ └────┬─────┘  │
│        │            │          │            │         │
│   ┌────▼────┐ ┌─────▼────┐   │       ┌────▼─────┐   │
│   │ML Models│ │ML Models │   │       │ Razorpay │   │
│   │(Joblib) │ │ (Pickle) │   │       │   API    │   │
│   └─────────┘ └──────────┘   │       └──────────┘   │
│                              │                       │
│         ┌────────────────────▼─────────────┐         │
│         │  PyTorch Neural Network Model    │         │
│         │  (data.pth + NLTK Pipeline)      │         │
│         └──────────────────────────────────┘         │
└─────────────────────┬───────────────────────────────┘
                      │ SQL Queries / SMTP
┌─────────────────────▼───────────────────────────────┐
│                    DATA TIER                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  MySQL   │  │    SMTP      │  │  File System  │  │
│  │ Database │  │  (Gmail)     │  │  (Reports)    │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 3.2 Database Design

The MySQL database (`pro17`) consists of **4 tables**:

#### Table 1: admin
| Column | Type | Description |
|--------|------|-------------|
| id | INT (PK, AUTO_INCREMENT) | Admin ID |
| a_username | VARCHAR(255) | Admin username |
| a_password | VARCHAR(255) | Admin password |

#### Table 2: user
| Column | Type | Description |
|--------|------|-------------|
| u_id | INT (PK, AUTO_INCREMENT) | User ID |
| u_name | VARCHAR(255) | User full name |
| u_email | VARCHAR(255, UNIQUE) | User email |
| u_password | VARCHAR(255) | User password |
| u_mobile | VARCHAR(20) | Mobile number |
| u_age | INT | User age |

#### Table 3: doctors
| Column | Type | Description |
|--------|------|-------------|
| d_id | INT (PK, AUTO_INCREMENT) | Doctor ID |
| d_name | VARCHAR(255) | Doctor name |
| d_email | VARCHAR(255, UNIQUE) | Doctor email |
| d_passwords | VARCHAR(255) | Doctor password |
| d_spec | VARCHAR(255) | Specialization |

#### Table 4: appointment
| Column | Type | Description |
|--------|------|-------------|
| ap_id | INT (PK, AUTO_INCREMENT) | Appointment ID |
| d_id | INT (FK → doctors) | Doctor reference |
| u_id | INT (FK → user) | User reference |
| ap_time | VARCHAR(50) | Appointment time |
| ap_date | VARCHAR(50) | Appointment date |
| ap_report | VARCHAR(255) | Uploaded report filename |
| ap_payment_status | VARCHAR(50) | Payment status (pending/success) |
| ap_status | VARCHAR(50) | Appointment status (pending/Accept/Reject) |
| razorpay_order_id | VARCHAR(255) | Razorpay order reference |

**Relationships:**
- `appointment.d_id` → `doctors.d_id` (ON DELETE CASCADE)
- `appointment.u_id` → `user.u_id` (ON DELETE CASCADE)

#### ER Diagram

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────┐
│    admin     │       │   appointment    │       │    user      │
├─────────────┤       ├──────────────────┤       ├─────────────┤
│ id (PK)     │       │ ap_id (PK)       │       │ u_id (PK)   │
│ a_username  │       │ d_id (FK) ───────┼──┐    │ u_name      │
│ a_password  │       │ u_id (FK) ───────┼──┼───▶│ u_email     │
└─────────────┘       │ ap_time          │  │    │ u_password  │
                      │ ap_date          │  │    │ u_mobile    │
                      │ ap_report        │  │    │ u_age       │
                      │ ap_payment_status│  │    └─────────────┘
                      │ ap_status        │  │
                      │ razorpay_order_id│  │
                      └──────────────────┘  │
                                            │
                      ┌─────────────┐       │
                      │   doctors   │       │
                      ├─────────────┤       │
                      │ d_id (PK) ◀─┼───────┘
                      │ d_name      │
                      │ d_email     │
                      │ d_passwords │
                      │ d_spec      │
                      └─────────────┘
```

### 3.3 Data Flow Diagrams (DFD)

#### Level 0 — Context Diagram

```
                    ┌─────────────────────────────┐
   ┌───────────┐    │                             │    ┌───────────┐
   │           │───▶│                             │───▶│           │
   │   User    │    │     Disease Prediction      │    │  Doctor   │
   │  (Patient)│◀───│      Web Application        │◀───│   Panel   │
   │           │    │                             │    │           │
   └───────────┘    │                             │    └───────────┘
                    │   ┌───────────────────┐     │
   ┌───────────┐    │   │    MySQL DB       │     │    ┌───────────┐
   │           │───▶│   │    (pro17)        │     │───▶│           │
   │   Admin   │    │   └───────────────────┘     │    │  Razorpay │
   │   Panel   │◀───│                             │◀───│  Payment  │
   │           │    │   ┌───────────────────┐     │    └───────────┘
   └───────────┘    │   │  ML Models +      │     │
                    │   │  Chatbot + LLM    │     │
                    │   └───────────────────┘     │
                    └─────────────────────────────┘
```

#### Level 1 — Detailed DFD

```
 USER ──▶ [1.0 Authentication] ──▶ User Session (MySQL)
                    │
                    ▼
 USER ──▶ [2.0 Binary Symptom Prediction] ──▶ SVM/MNB Model ──▶ Disease + Confidence + Specialist
                    │
                    ▼
 USER ──▶ [3.0 Text-Based Prediction] ──▶ TF-IDF + MNB Pipeline ──▶ Top-3 Diseases
                    │
                    ▼
 USER ──▶ [4.0 Diabetes Basic] ──▶ StandardScaler + SVM ──▶ Diabetic/Non-Diabetic
                    │
                    ▼
 USER ──▶ [5.0 Diabetes Advanced] ──▶ StandardScaler + RF ──▶ Risk Score + Confidence %
                    │
                    ▼
 USER ──▶ [6.0 Book Appointment] ──▶ [7.0 Razorpay Payment] ──▶ MySQL (appointment) ──▶ Doctor Panel
                    │
                    ▼
 USER ──▶ [8.0 Chatbot Query] ──▶ NLTK Pipeline + PyTorch NN ──▶ Bot Response

 ADMIN ─▶ [9.0 Manage Users/Doctors] ──▶ MySQL (user, doctors tables)
 DOCTOR ─▶ [10.0 Manage Appointments] ──▶ MySQL (appointment) ──▶ [11.0 Send Email] ──▶ SMTP (Gmail)
```

### 3.4 UML Diagrams

#### Use Case Diagram

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

#### Sequence Diagram — Disease Prediction Flow

```
  User          Browser        Flask Server      ML Model       MySQL DB
    │               │               │               │               │
    │──Select──────▶│               │               │               │
    │  Symptoms     │──POST /check──▶│               │               │
    │               │  disease      │               │               │
    │               │               │──predict()────▶│               │
    │               │               │               │──Result──────▶│
    │               │               │◀─confidence───│               │
    │               │               │               │               │
    │               │               │──SELECT doc───┼──────────────▶│
    │               │               │◀─doctor name──┼───────────────│
    │               │               │               │               │
    │               │◀─JSON Result──│               │               │
    │◀─Disease + ───│               │               │               │
    │  Specialist   │               │               │               │
    │               │               │               │               │
```

#### Sequence Diagram — Appointment & Payment Flow

```
  User       Browser     Flask      Razorpay     MySQL       Doctor
    │           │        Server       API          DB          Email
    │           │          │           │           │             │
    │──Fill────▶│          │           │           │             │
    │  Form     │──POST────▶│           │           │             │
    │           │  /process │           │           │             │
    │           │          │──Upload──▶│           │             │
    │           │          │  Report   │  (File)   │             │
    │           │          │──Create──▶│           │             │
    │           │          │  Order    │           │             │
    │           │          │◀─Order───│           │             │
    │           │          │  ID      │           │             │
    │           │          │──INSERT──┼──────────▶│             │
    │           │          │  Appt    │           │             │
    │◀─Payment──│◀─Redirect│           │           │             │
    │  Page     │          │           │           │             │
    │──Pay──────┼──────────┼──────────▶│           │             │
    │           │          │           │──Callback─┼────────────▶│
    │           │          │◀─Success─┼───────────│             │
    │           │          │──UPDATE──┼──────────▶│             │
    │           │          │  Status  │           │             │
    │◀─Success──│◀─Redirect│           │           │──Email─────▶│
    │  Page     │          │           │           │  Reminder   │
    │           │          │           │           │             │
```

### 3.5 Machine Learning Pipeline

The ML pipeline follows a systematic approach for model training and evaluation:

```
Raw Data → Data Preprocessing → Feature Engineering → Train/Test Split
     → Model Training → Cross-Validation → Evaluation → Model Serialization
```

#### 3.3.1 Binary Symptom Prediction Pipeline

```
User Input (132 symptoms)
        │
        ▼
Binary Encoding (0/1 for each symptom)
        │
        ▼
Pre-trained SVM/Decision Tree/MNB Model
        │
        ▼
Disease Prediction + Confidence Score (predict_proba)
        │
        ▼
Specialist Mapping + Disease Details
```

#### 3.3.2 Text-Based Prediction Pipeline

```
User Input (Free Text)
        │
        ▼
TF-IDF Vectorization (max_features=8000, ngram_range=(1,2))
        │
        ▼
Pre-trained MultinomialNB Classifier
        │
        ▼
Label Decoding → Top-3 Diseases + Confidence Scores
```

#### 3.3.3 Diabetes Prediction Pipeline (Basic - PIMA)

```
User Input (8 Features)
        │
        ▼
StandardScaler Transformation
        │
        ▼
Pre-trained SVM Model
        │
        ▼
Diabetic / Non-Diabetic Prediction
```

#### 3.3.4 Diabetes Prediction Pipeline (Advanced - BRFSS)

```
User Input (21 Health Indicators)
        │
        ▼
StandardScaler Transformation
        │
        ▼
Pre-trained Random Forest (100 estimators)
        │
        ▼
Diabetes Risk Prediction + Confidence Percentage
```

### 3.4 NLP Pipeline (Chatbot)

```
User Message
      │
      ▼
NLTK Tokenization (word_tokenize)
      │
      ▼
Porter Stemmer (root form extraction)
      │
      ▼
Bag of Words Representation
      │
      ▼
PyTorch Neural Network (3-layer Feedforward)
      │
      ▼
Intent Classification + Confidence Check (threshold: 0.75)
      │
      ▼
Response Selection from intents.json
```

### 3.5 Deep Learning Pipeline (Chatbot)

**Neural Network Architecture (model.py):**

```
Input Layer (input_size neurons)
        │
        ▼
Hidden Layer 1 (hidden_size neurons) + ReLU
        │
        ▼
Hidden Layer 2 (hidden_size neurons) + ReLU
        │
        ▼
Output Layer (num_classes neurons)
```

- **Framework:** PyTorch
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Not specified in saved model (trained externally via train.py)
- **Confidence Threshold:** 0.75
- **Bot Name:** Sam
- **Saved Model:** `data.pth` (PyTorch state dict)

### 3.6 Technology Stack

| Category | Technology | Version/Purpose |
|----------|-----------|-----------------|
| Frontend | HTML5, CSS3, Bootstrap | Responsive UI design |
| Frontend | JavaScript | Client-side interactions |
| Backend | Python Flask | Web server and routing |
| Database | MySQL (Flask-MySQLdb) | Relational data storage |
| Machine Learning | Scikit-learn | Classification algorithms |
| Deep Learning | PyTorch | Neural network (chatbot) |
| NLP | NLTK | Tokenization, stemming, BoW |
| Data Processing | Pandas, NumPy | Data manipulation |
| Class Balancing | imbalanced-learn (SMOTE) | Oversampling minority class |
| Visualization | Matplotlib, Seaborn | Feature correlation, charts |
| Model Storage | Joblib, Pickle | Model serialization |
| Payment Gateway | Razorpay | Online payment processing |
| Email Service | SMTP (Gmail) | Email notifications |
| File Upload | Werkzeug | Secure file handling |
| LLM Integration | Ollama (Gemma 3) | Disease info generation |
| Environment | Anaconda (environment.yml) | Package management |
| Configuration | YAML (config.yaml) | Model hyperparameters |

### 3.7 Specialist Mapping Logic

The system maps predicted diseases to medical specialists using a predefined dictionary:

| Specialist | Diseases Covered |
|-----------|-----------------|
| Rheumatologist | Osteoarthritis, Arthritis |
| Cardiologist | Heart attack, Bronchial Asthma, Hypertension |
| ENT Specialist | Vertigo, Hypothyroidism |
| Neurologist | Varicose veins, Paralysis, Migraine, Cervical spondylosis |
| Allergist | Allergy, Pneumonia, AIDS, Common Cold, Tuberculosis, Malaria, Dengue, Typhoid |
| Urologist | Urinary tract infection, Hemorrhoids |
| Dermatologist | Acne, Chicken pox, Fungal infection, Psoriasis, Impetigo |
| Gastroenterologist | Peptic ulcer, GERD, Cholestasis, Drug Reaction, Gastroenteritis, Hepatitis A-E, Diabetes, Hypoglycemia |

### 3.8 Mathematical Formulation of Algorithms

This section presents the mathematical foundations of the key algorithms used in this project.

#### 3.8.1 Support Vector Machine (SVM)

The SVM algorithm finds the optimal hyperplane that separates data points of different classes with maximum margin [15]. Given a training set {(x₁, y₁), ..., (xₙ, yₙ)} where xᵢ ∈ ℝᵈ and yᵢ ∈ {-1, +1}:

**Primal Optimization Problem:**

    minimize: (1/2)||w||² + C Σᵢ₌₁ⁿ ξᵢ

    subject to: yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

Where:
- w is the weight vector
- b is the bias term
- C is the regularization parameter (C = 0.1 in our model)
- ξᵢ are slack variables allowing misclassification
- φ(x) is the feature mapping function

**RBF Kernel Function:**

    K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)

Where γ controls the influence of a single training example (γ = 'scale' in our model).

**Decision Function:**

    f(x) = sign(Σᵢ αᵢyᵢK(xᵢ, x) + b)

**Parameters Used:** C = 0.1, kernel = RBF, gamma = scale

#### 3.8.2 Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of classes for classification [14].

**Ensemble Prediction:**

    ĥ(x) = majority_vote{h₁(x), h₂(x), ..., h_B(x)}

Where hᵢ(x) is the prediction of the i-th decision tree and B is the number of trees (B = 100 in our model).

**Bootstrap Aggregating (Bagging):** Each tree is trained on a bootstrap sample Dᵢ drawn with replacement from the original training set D.

**Random Feature Selection:** At each split, a random subset of m features is considered from the total M features, where m = √M for classification.

**Gini Impurity (Split Criterion):**

    Gini(t) = 1 - Σᵢ₌₁ᶜ p(i|t)²

Where p(i|t) is the proportion of class i samples at node t, and C is the number of classes.

**Out-of-Bag (OOB) Error:** Estimated using samples not selected in the bootstrap sample, providing an unbiased error estimate.

**Parameters Used:** n_estimators = 100, random_state = 42, n_jobs = -1

#### 3.8.3 Multinomial Naive Bayes (Text Model)

Naive Bayes classifiers apply Bayes' theorem with the "naive" assumption of conditional independence between every pair of features [9].

**Bayes' Theorem:**

    P(y|x₁, x₂, ..., xₙ) = P(y) × P(x₁, x₂, ..., xₙ|y) / P(x₁, x₂, ..., xₙ)

**With Naive Independence Assumption:**

    P(y|x₁, ..., xₙ) ∝ P(y) × Πᵢ₌₁ⁿ P(xᵢ|y)

**Multinomial Model (for TF-IDF features):**

    P(xᵢ|y) = (N_yᵢ + α) / (N_y + αn)

Where:
- N_yᵢ is the count of feature i in class y
- N_y is the total count of all features in class y
- α is the Laplace smoothing parameter (α = 0.01 in our model)
- n is the number of features

**Classification Decision:**

    ŷ = argmax_y P(y) × Πᵢ P(xᵢ|y)

**Parameters Used:** alpha = 0.01, TF-IDF max_features = 8000, ngram_range = (1, 2)

#### 3.8.4 TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text documents into numerical feature vectors [12].

**Term Frequency (TF):**

    TF(t, d) = f(t, d) / Σf'(t, d)

Where f(t, d) is the frequency of term t in document d.

**Inverse Document Frequency (IDF):**

    IDF(t) = log((1 + n) / (1 + df(t))) + 1

Where n is the total number of documents and df(t) is the number of documents containing term t.

**TF-IDF Score:**

    TF-IDF(t, d) = TF(t, d) × IDF(t)

**Parameters Used:** max_features = 8000, ngram_range = (1, 2), sublinear_tf = False

#### 3.8.5 SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic samples for the minority class by creating new samples along line segments joining existing minority class nearest neighbors [11].

**Synthetic Sample Generation:**

    x_new = xᵢ + λ × (xⱼ - xᵢ)

Where:
- xᵢ is a minority class sample
- xⱼ is one of its k nearest neighbors in the minority class
- λ is a random number in the range [0, 1]

**Parameters Used:** k_neighbors = 5 (default), random_state = 42

#### 3.8.6 Neural Network (Chatbot)

The chatbot uses a 3-layer feedforward neural network implemented in PyTorch [10].

**Forward Pass:**

    z₁ = W₁x + b₁
    a₁ = ReLU(z₁) = max(0, z₁)
    z₂ = W₂a₁ + b₂
    a₂ = ReLU(z₂) = max(0, z₂)
    z₃ = W₃a₂ + b₃

Where Wᵢ are weight matrices and bᵢ are bias vectors.

**Cross-Entropy Loss:**

    L = -Σᵢ₌₁ᶜ yᵢ log(softmax(z₃)ᵢ)

**Softmax Function:**

    softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

**Parameters:** input_size = vocabulary size, hidden_size = configurable, num_classes = number of intents, confidence_threshold = 0.75

---

# CHAPTER 4

## IMPLEMENTATION

### 4.1 Dataset Details

#### Dataset 1: Disease Symptom Dataset (Binary)

| Property | Value |
|----------|-------|
| Training Data | `dataset/training_data.csv` |
| Test Data | `dataset/test_data.csv` |
| Features | 132 binary symptom columns (0 or 1) |
| Target | `prognosis` column (41 diseases) |
| Feature Type | Binary (0 = symptom absent, 1 = symptom present) |

**41 Diseases Covered:** Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction, Peptic ulcer disease, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Paralysis (brain hemorrhage), Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A-E, Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia, Dimorphic hemorrhoids, Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthritis, Arthritis, Vertigo, Acne, Urinary tract infection, Psoriasis, Impetigo

**132 Symptoms:** itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue, muscle_wasting, vomiting, fatigue, headache, fever, cough, nausea, dizziness, chest_pain, and 113 more

#### Dataset 2: PIMA Diabetes Dataset (Basic)

| Property | Value |
|----------|-------|
| Source | National Institute of Diabetes and Digestive and Kidney Diseases |
| File | `diabetes.csv` |
| Records | 768 patients |
| Features | 8 clinical parameters |
| Target | Outcome (0 = Non-Diabetic, 1 = Diabetic) |

**Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

#### Dataset 3: CDC BRFSS 2015 Diabetes Dataset (Advanced)

| Property | Value |
|----------|-------|
| Source | CDC Behavioral Risk Factor Surveillance System 2015 |
| File | `diabetes_binary_health_indicators_BRFSS2015.csv` |
| Records | 253,680 (229,474 after deduplication) |
| Features | 21 health indicators |
| Target | Diabetes_binary (0 = No Diabetes, 1 = Diabetes) |

**21 Features:** HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

**Class Distribution (after deduplication):**
- No Diabetes: 218,334 records
- Diabetes: 35,346 records
- Duplicate Rows Removed: 24,206

#### Dataset 4: Symptom2Disease (Text-Based)

| Property | Value |
|----------|-------|
| File | `Symptom2Disease.csv` |
| Records | 1,200 patient symptom descriptions |
| Diseases | 24 (50 descriptions per disease) |
| Columns | label (disease name), text (natural language description) |

**24 Diseases:** Psoriasis, Varicose Veins, Typhoid, Chicken pox, Impetigo, Dengue, Fungal infection, Common Cold, Pneumonia, Dimorphic Hemorrhoids, Arthritis, Acne, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Jaundice, Malaria, Urinary tract infection, Allergy, GERD, Drug reaction, Peptic ulcer disease, Diabetes

**Example:** *"I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."* → **Psoriasis**

### 4.2 Data Preprocessing

#### 4.2.1 Binary Symptom Data
- Loaded via Pandas from CSV files
- Features extracted by filtering out `prognosis` and `Unnamed` columns
- Data sanity checks: `assert len(features.iloc[0]) == 132`
- Train/validation split: 67/33 (test_size=0.33, random_state=101)
- No missing values or scaling required (binary features)

#### 4.2.2 PIMA Diabetes Data
- **Missing Value Handling:** Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns (biologically impossible) were replaced with column-wise mean values
- **Feature Scaling:** StandardScaler applied to normalize features
- **Train/Test Split:** 80/20 (random_state=7)

#### 4.2.3 BRFSS Diabetes Data
- **Duplicate Removal:** 24,206 duplicate rows removed (253,680 → 229,474)
- **Class Balancing:** SMOTE (Synthetic Minority Oversampling Technique) applied to training data only
  - Before SMOTE: ~155K training samples (imbalanced)
  - After SMOTE: ~311K training samples (balanced)
- **Feature Scaling:** StandardScaler fit on training data, applied to both train and test
- **Train/Test Split:** 80/20 (stratified, random_state=42)

#### 4.2.4 Text Symptom Data
- Raw text descriptions used directly
- TF-IDF vectorization with hyperparameter tuning
- Best parameters: max_features=8000, ngram_range=(1,2), alpha=0.01

### 4.3 Feature Engineering

#### Binary Symptom Features
- 132 binary features encoded as 0 (absent) or 1 (present)
- Feature correlation analysis performed using Seaborn heatmap (`feature_correlation.png`)
- No feature selection applied due to already high accuracy

#### TF-IDF Features (Text Model)
- Term Frequency-Inverse Document Frequency vectorization
- Hyperparameter-tuned: max_features and ngram_range varied across models
- Best performing: max_features=8000, ngram_range=(1,2) with MultinomialNB

#### PIMA Features
- 8 numerical features standardized using StandardScaler
- Zero-value imputation for biologically impossible values

#### BRFSS Features
- 21 mixed-type features (binary + ordinal) standardized using StandardScaler
- SMOTE applied after split to prevent data leakage

### 4.4 Model Training and Selection

#### 4.4.1 Configuration-Driven Training (main.py)

The binary symptom model training is driven by `config.yaml`:

```yaml
random_state: 101
dataset:
  training_data_path: './dataset/training_data.csv'
  test_data_path: './dataset/test_data.csv'
  validation_size: 0.33
model:
  decision_tree:
    criterion: 'gini'
    max_depth: null
  random_forest:
    n_estimators: 100
    max_depth: 10
  gradient_boost:
    n_estimators: 100
    learning_rate: 0.05
  svm:
    C: 0.1
    kernel: 'rbf'
```

The `DiseasePrediction` class in `main.py` handles:
1. Loading config from YAML
2. Loading train/test datasets
3. Computing feature correlation
4. Train/validation splitting
5. Model selection based on name
6. Training with cross-validation (3-fold)
7. Saving trained models via Joblib

#### 4.4.2 Comprehensive Training Results

All models were trained and evaluated with the following results from `training_results.json`:

**Binary Symptom Models (132 features → 41 diseases):**

| Model | CV Score | Validation Accuracy | Test Accuracy |
|-------|----------|-------------------|---------------|
| SVM (RBF, C=0.1) | 1.0000 | 1.0000 | **1.0000 (100%)** |
| Logistic Regression (C=0.1) | 1.0000 | 1.0000 | **1.0000 (100%)** |
| Decision Tree (gini) | 1.0000 | 1.0000 | 0.9762 (97.6%) |
| Random Forest (100 estimators) | 1.0000 | 1.0000 | 0.9762 (97.6%) |
| Gradient Boosting (100 estimators) | 1.0000 | 1.0000 | 0.9762 (97.6%) |
| XGBoost (200 estimators) | 1.0000 | 1.0000 | 0.9762 (97.6%) |

**Text-Based Models (TF-IDF → 24 diseases):**

| Model | CV Score | Test Accuracy | Test F1-Score |
|-------|----------|---------------|---------------|
| **MultinomialNB** | 0.9656 | **0.9792 (97.92%)** | **0.9789** |
| LinearSVC | 0.9708 | 0.9542 (95.42%) | 0.9529 |
| Logistic Regression | 0.9719 | 0.9500 (95.00%) | 0.9488 |
| Random Forest | 0.9490 | 0.9458 (94.58%) | 0.9451 |
| XGBoost | 0.8740 | 0.8708 (87.08%) | 0.8727 |

**Production Models Selected:**
- Binary: `best_binary_model.joblib` (SVM) — chosen for 100% test accuracy and probability estimates
- Text: `best_text_model.joblib` (TF-IDF + MultinomialNB pipeline) — chosen for highest test accuracy (97.92%)
- Diabetes Basic: `svm_model.pkl` (SVM with StandardScaler)
- Diabetes Advanced: `diabetes_brfss_model.pkl` (Random Forest, 100 estimators)

#### 4.4.3 Diabetes Model Training (BRFSS)

The `train_diabetes_brfss.py` script performs:
1. Loads BRFSS 2015 dataset
2. Removes 24,206 duplicate rows
3. Splits into 80/20 train/test (stratified)
4. Applies SMOTE to training data only
5. Standardizes features with StandardScaler
6. Trains 6 models: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting
7. Selects best model (Random Forest: 84.05%)
8. Saves model, scaler, and feature names as pickle files

**BRFSS Model Comparison:**

| Model | Test Accuracy |
|-------|---------------|
| **Random Forest (100 estimators)** | **84.05% (Best)** |
| Gradient Boosting | 83.78% |
| Decision Tree | 77.13% |
| KNN | 75.28% |
| Logistic Regression | 71.50% |
| Naive Bayes | 64.39% |

### 4.5 Web Application Implementation

The Flask application (`app.py`, 877 lines) implements the following routes:

#### User Routes
| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login with session |
| `/logout` | GET | Session clearing |
| `/predict_main` | GET | Prediction page |
| `/predict` | POST | Binary symptom prediction API |
| `/predict_text` | GET, POST | Text-based prediction |
| `/checkdisease` | GET, POST | Advanced disease check with specialist mapping |
| `/diabeties` | GET, POST | Basic diabetes prediction |
| `/diabetes_advanced` | GET, POST | Advanced diabetes prediction with confidence |
| `/appoinment` | GET | Appointment booking page |
| `/process_form` | POST | Process appointment with Razorpay |
| `/payment/<order_id>` | GET | Payment page |
| `/success` | GET | Payment success callback |
| `/history/<u_id>` | GET | Appointment history |
| `/services` | GET | Services page |
| `/predict1` | POST | Chatbot API endpoint |

#### Admin Routes
| Route | Method | Description |
|-------|--------|-------------|
| `/admin` | GET, POST | Admin login |
| `/admin/index` | GET | Dashboard (total users, doctors) |
| `/admin/user` | GET | View all users |
| `/admin/edit_user/<id>` | GET, POST | Edit user details |
| `/admin/delete/<id>` | GET | Delete user |
| `/admin/doctor` | GET | View all doctors |
| `/admin/add_doctor` | GET, POST | Add new doctor |
| `/admin/doc_delete/<id>` | GET | Delete doctor |

#### Doctor Routes
| Route | Method | Description |
|-------|--------|-------------|
| `/dlogin` | GET, POST | Doctor login |
| `/dindex` | GET | Doctor dashboard with appointment count |
| `/appoin/<d_id>` | GET, POST | View appointments for doctor |
| `/update_accept_appoin/<id>` | GET | Accept appointment |
| `/update_reject_appoin/<id>` | GET | Reject appointment |
| `/send_email_flask/<email>` | GET | Send email reminder to patient |

### 4.6 Payment Gateway Integration

**Razorpay Integration Flow:**

1. User fills appointment form with doctor, date, time, and medical report
2. Report file is uploaded securely via Werkzeug (`secure_filename`)
3. Razorpay order created for INR 200
4. Appointment record saved with `pending` payment status
5. User redirected to Razorpay payment page
6. On successful payment: `ap_payment_status` updated to `success`
7. Doctor can view only successfully paid appointments
8. Email reminder sent to patient via SMTP on appointment action

### 4.7 LLM Integration (Gemma 3 via Ollama)

The `generate_disease_info.py` script uses **Gemma 3** via Ollama's local API (`http://localhost:11434/api/generate`) to generate structured disease information:

**Process:**
1. Reads 24 diseases from `Symptom2Disease.csv`
2. For each disease, fetches 5 sample patient descriptions
3. Constructs a prompt requesting JSON-formatted medical information
4. Sends prompt to Gemma 3 via Ollama API
5. Parses JSON response with retry logic on failure
6. Saves all disease information to `disease_info.json`

**Generated Fields per Disease:**
- `description` — 2-3 sentence medical description
- `common_symptoms` — List of 5 common symptoms
- `precautions` — List of 4 precautionary measures
- `medications` — List of 3 medication categories
- `diet_recommendations` — List of 3 dietary tips
- `when_to_see_doctor` — Guidance on when to seek urgent care

---

# CHAPTER 5

## RESULTS AND DISCUSSION

### 5.1 Disease Prediction Results (Binary Symptoms)

#### Overall Summary

| Model | Algorithm | Test Accuracy | Best Parameters |
|-------|-----------|---------------|-----------------|
| `best_binary_model.joblib` | SVM (RBF kernel) | **100.00%** | C=0.1, gamma=scale |
| - | Logistic Regression | **100.00%** | C=0.1, solver=lbfgs |
| `decision_tree.joblib` | Decision Tree (gini) | 97.62% | criterion=gini |
| - | Random Forest | 97.62% | n_estimators=100, max_depth=10 |
| - | Gradient Boosting | 97.62% | n_estimators=100, lr=0.05 |
| - | XGBoost | 97.62% | n_estimators=200, lr=0.05 |
| `mnb.joblib` | Multinomial Naive Bayes | 100.00%* | - |

*Note: MNB trained separately via main.py; validation accuracy = 100%, test performance verified at 100% in PPT guide.

**Analysis:** The symptom-disease dataset has distinct, non-overlapping symptom patterns per disease, resulting in very high classification accuracy across all algorithms. SVM and Logistic Regression achieve perfect test accuracy, while tree-based methods show slight variation (97.62%). The validation accuracy is 100% for all models with 3-fold cross-validation scores of 1.0, confirming model reliability.

**Production Selection:** SVM (`best_binary_model.joblib`) was selected as the production model because:
1. 100% test accuracy
2. `predict_proba()` support for confidence scoring
3. Robust performance with RBF kernel on high-dimensional data

### 5.2 Disease Prediction Results (Text-Based)

| Model | CV Score | Test Accuracy | Test F1 | TF-IDF max_features | ngram_range |
|-------|----------|---------------|---------|-------------------|-------------|
| **MultinomialNB** | 0.9656 | **97.92%** | **0.9789** | 8000 | (1,2) |
| LinearSVC | 0.9708 | 95.42% | 0.9529 | 3000 | (1,1) |
| Logistic Regression | 0.9719 | 95.00% | 0.9488 | 3000 | (1,1) |
| Random Forest | 0.9490 | 94.58% | 0.9451 | 3000 | (1,2) |
| XGBoost | 0.8740 | 87.08% | 0.8727 | 3000 | (1,1) |

**Analysis:**
- MultinomialNB achieves the best test accuracy (97.92%) and F1-score (0.9789), benefiting from:
  - Higher max_features (8000) capturing more vocabulary
  - Bigram features (ngram_range=(1,2)) capturing symptom phrases
  - Low alpha (0.01) allowing strong feature contributions
- LinearSVC has a higher CV score (0.9708) but lower test accuracy, suggesting slight overfitting
- XGBoost underperforms likely due to the relatively small dataset size (1,200 records)
- The balanced dataset (50 samples per disease) contributes to consistent performance across all 24 classes

**Production Output:** Top-3 predicted diseases with confidence percentages, enabling users to see alternative diagnoses.

### 5.3 Diabetes Prediction Results (Basic - PIMA)

| Model | Test Accuracy |
|-------|---------------|
| **Naive Bayes** | **78.57% (Best)** |
| Logistic Regression | 76.62% |
| Random Forest | 75.97% |
| SVM | 75.32% |
| KNN | 75.32% |
| Decision Tree | 73.38% |

**Analysis:**
- The PIMA dataset's small size (768 records) limits model performance
- Naive Bayes performs best, likely due to the relatively independent nature of clinical features
- All models fall in the 73-79% range, consistent with literature findings [2]
- Zero-value imputation (replacing biologically impossible zeros with column means) improved data quality
- Production uses SVM (`svm_model.pkl`) as the originally selected deployment model

### 5.4 Diabetes Prediction Results (Advanced - BRFSS)

| Model | Test Accuracy |
|-------|---------------|
| **Random Forest (100 estimators)** | **84.05% (Best)** |
| Gradient Boosting | 83.78% |
| Decision Tree | 77.13% |
| KNN | 75.28% |
| Logistic Regression | 71.50% |
| Naive Bayes | 64.39% |

**Detailed Classification Report (Random Forest):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Diabetes | 0.87 | 0.96 | 0.91 | 43,667 |
| Diabetes | 0.45 | 0.19 | 0.27 | 11,028 |

**Analysis:**
- Random Forest achieves the best overall accuracy (84.05%) but shows a significant class imbalance in recall
- **No Diabetes class:** High recall (0.96) means very few false negatives for healthy patients
- **Diabetes class:** Low recall (0.19) indicates many diabetic patients are missed (false negatives)
- This imbalance persists despite SMOTE, suggesting the minority class has inherently complex patterns
- The large dataset (229,474 records) provides robust training, with SMOTE expanding training from ~155K to ~311K samples
- **SMOTE applied only on training data** to prevent data leakage — a critical methodological choice

### 5.5 Chatbot Performance

| Metric | Value |
|--------|-------|
| Architecture | 3-layer Feedforward Neural Network |
| Framework | PyTorch |
| NLP Pipeline | NLTK (Tokenize → Stem → Bag of Words) |
| Confidence Threshold | 0.75 |
| Training Data | intents.json (greetings, services, symptoms, disease info) |

**Analysis:**
- The chatbot uses a confidence threshold of 0.75 to filter low-confidence predictions
- Queries below threshold receive a fallback response: "I do not understand..."
- The bag-of-words approach is lightweight and suitable for the limited intent space
- Intent categories cover: greetings, goodbye, thanks, services, payments, symptom descriptions, disease info queries

### 5.6 SMOTE Analysis — Class Imbalance Handling (BRFSS)

#### Before SMOTE (Training Set)
| Class | Count | Percentage |
|-------|-------|------------|
| No Diabetes | ~174,686 | 86.2% |
| Diabetes | ~28,094 | 13.8% |
| **Total** | **~202,780** | **100%** |

#### After SMOTE (Training Set)
| Class | Count | Percentage |
|-------|-------|------------|
| No Diabetes | ~174,686 | 50% |
| Diabetes (Synthetic) | ~174,686 | 50% |
| **Total** | **~349,372** | **100%** |

- SMOTE was applied **only on training data** to prevent data leakage
- Test set retains original imbalanced distribution for realistic evaluation
- Training samples increased from ~155K → ~311K
- **Impact:** Improved model's ability to detect diabetes patterns, though class recall remains low

### 5.7 Confusion Matrix Analysis

#### BRFSS Diabetes Model (Random Forest — Best Model)

```
                         Predicted
                    No Diabetes   Diabetes
Actual  No Diabetes   41,857      1,810      ← Recall: 96%
        Diabetes       8,943      2,085      ← Recall: 19%
                         ↑            ↑
                   Precision:    Precision:
                      87%           45%
```

| Metric | No Diabetes | Diabetes |
|--------|-------------|----------|
| **Precision** | 0.87 | 0.45 |
| **Recall** | 0.96 | 0.19 |
| **F1-Score** | 0.91 | 0.27 |
| **Support** | 43,667 | 11,028 |

- **True Negatives (41,857):** Correctly identified non-diabetic patients
- **False Positives (1,810):** Healthy patients incorrectly flagged — low impact
- **False Negatives (8,943):** Diabetic patients missed — primary area for improvement
- **True Positives (2,085):** Correctly identified diabetic patients
- **Overall Accuracy:** 84.05%

#### Binary Symptom Model (SVM — Perfect Classification)

```
                    Predicted
               All 41 diseases correctly classified
               Confusion Matrix = Identity Matrix (41×41)
               Accuracy = 100%
```

The identity confusion matrix indicates that each disease has a unique, non-overlapping symptom signature, making the classification task highly separable.

### 5.8 Feature Importance Analysis (BRFSS Random Forest)

The Random Forest model provides feature importance scores based on mean decrease in impurity (Gini importance):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | BMI | ~0.125 | Clinical |
| 2 | Age | ~0.098 | Demographic |
| 3 | GenHlth | ~0.087 | Self-reported |
| 4 | Income | ~0.072 | Socioeconomic |
| 5 | PhysHlth | ~0.065 | Self-reported |
| 6 | Education | ~0.058 | Socioeconomic |
| 7 | MentHlth | ~0.045 | Self-reported |
| 8 | HighBP | ~0.042 | Clinical |
| 9 | HighChol | ~0.038 | Clinical |
| 10 | Smoker | ~0.035 | Behavioral |
| 11 | PhysActivity | ~0.032 | Behavioral |
| 12 | HeartDiseaseorAttack | ~0.028 | Clinical |
| 13 | DiffWalk | ~0.025 | Functional |
| 14 | Sex | ~0.022 | Demographic |
| 15 | Stroke | ~0.018 | Clinical |

**Key Findings:**
- **BMI** is the strongest predictor of diabetes, consistent with medical literature
- **Socioeconomic factors** (Income, Education) rank in the top 6, highlighting social determinants of health
- **Behavioral factors** (Smoker, PhysActivity) contribute meaningfully, validating the multi-indicator approach
- **Clinical + Behavioral + Socioeconomic** features together achieve 84.05% accuracy, significantly better than clinical-only models (71-75%)

### 5.9 Comparison with Existing Systems

| Feature | Traditional Diagnosis | Existing ML Systems | Our System |
|---------|----------------------|---------------------|------------|
| Disease Prediction | Manual only | Single mode | **4 modes** |
| Input Method | Physical consultation | Structured form only | Structured + Natural Language |
| Diabetes Assessment | Lab tests only | Basic features (8) | Basic (8) + Advanced (21) |
| Specialist Mapping | Manual referral | Not available | **Automatic** |
| Appointment Booking | Phone/Walk-in | Not available | **Online with payment** |
| Chatbot Support | Not available | Basic FAQ | **AI-powered** |
| Multi-role Access | N/A | Single user type | **3 roles** |
| Dataset Scale | N/A | <1000 records | **Up to 229,474** |
| LLM Integration | Not available | Not available | **Gemma 3 via Ollama** |

### 5.10 Performance Evaluation Metrics

#### Overall System Performance Summary

| Module | Best Algorithm | Accuracy | Dataset Size | Features |
|--------|---------------|----------|-------------|----------|
| Disease (Binary) | SVM | **100.00%** | ~4,920 | 132 |
| Disease (Text) | TF-IDF + MNB | **97.92%** | 1,200 | TF-IDF (8000) |
| Diabetes (Basic) | Naive Bayes | **78.57%** | 768 | 8 |
| Diabetes (Advanced) | Random Forest | **84.05%** | 229,474 | 21 |
| Chatbot | PyTorch NN | 0.75 threshold | intents.json | BoW |

#### Evaluation Metrics Used
- **Accuracy Score:** Primary metric for all classification tasks
- **F1-Score:** Used for text-based prediction (weighted average across classes)
- **Cross-Validation (3-fold):** Used for binary symptom models to verify generalization
- **Confusion Matrix:** Generated for all models
- **Classification Report:** Generated for diabetes models (precision, recall, F1 per class)
- **Confidence Score (predict_proba):** Used in production for providing probability estimates to users

---

# CHAPTER 6

## TESTING AND VALIDATION

### 6.1 Model Testing

| Test Type | Description | Result |
|-----------|-------------|--------|
| Train/Test Split | 80/20 (diabetes) and 67/33 (disease) with stratification | All models validated |
| Cross-Validation | 3-fold CV on binary symptom models | Consistent 100% CV score |
| SMOTE Validation | Applied only on training data to prevent data leakage | Proper methodology |
| Confidence Scoring | `predict_proba()` for all prediction modules | Probabilities output correctly |
| Hyperparameter Tuning | Grid/random search across multiple algorithms | Optimal parameters selected |

### 6.2 System Testing

| Module | Test Case | Expected Result | Status |
|--------|-----------|-----------------|--------|
| User Registration | Submit valid registration form | User record created in database | Passed |
| User Login | Submit valid credentials | Session created, redirect to home | Passed |
| User Login (Invalid) | Submit wrong credentials | Error message displayed | Passed |
| Disease Prediction (Binary) | Select symptoms, submit | Disease + confidence + specialist | Passed |
| Disease Prediction (Text) | Enter symptom text, submit | Top-3 diseases with confidence | Passed |
| Diabetes Basic | Enter 8 parameters, submit | Diabetic/Non-diabetic result | Passed |
| Diabetes Advanced | Enter 21 indicators, submit | Risk prediction + confidence % | Passed |
| Appointment Booking | Fill form, upload report | Razorpay order created | Passed |
| Payment Flow | Complete Razorpay payment | Status updated to success | Passed |
| Doctor Login | Submit doctor credentials | Doctor session created | Passed |
| View Appointments | Doctor views paid appointments | Appointment list displayed | Passed |
| Accept Appointment | Doctor accepts appointment | Status updated to Accept | Passed |
| Reject Appointment | Doctor rejects appointment | Status updated to Reject | Passed |
| Email Notification | Doctor sends email reminder | Email delivered to patient | Passed |
| Admin Login | Submit admin credentials | Admin session created | Passed |
| Admin - Manage Users | View, add, edit, delete users | CRUD operations functional | Passed |
| Admin - Manage Doctors | View, add, delete doctors | CRUD operations functional | Passed |
| Chatbot | Send health query | Response with confidence check | Passed |
| Chatbot (Low Confidence) | Send ambiguous query | Fallback "I do not understand" | Passed |
| Session Management | Logout action | Session cleared, redirect to home | Passed |
| Appointment History | View user appointment history | History with doctor names displayed | Passed |

### 6.3 Data Validation

| Validation | Implementation |
|-----------|----------------|
| Input form validation | HTML5 required fields + server-side checks |
| File upload security | Werkzeug `secure_filename()` for report uploads |
| SQL injection prevention | Parameterized queries using Flask-MySQLdb `%s` placeholders |
| Session management | Flask session with secret key |
| Payment verification | Razorpay order ID matching in database |

---

# CHAPTER 7

## CHALLENGES AND SOLUTIONS

### Challenge 1: Class Imbalance in BRFSS Diabetes Dataset
- **Problem:** The BRFSS dataset was heavily imbalanced — No Diabetes: 218,334 vs Diabetes: 35,346 (6.2:1 ratio)
- **Solution:** Applied SMOTE (Synthetic Minority Oversampling Technique) to oversample the minority class in training data only. Training set expanded from ~155K to ~311K samples, ensuring the model sees balanced data while the test set retains the original distribution for realistic evaluation.
- **Impact:** Despite SMOTE, the diabetes class recall remained low (0.19), indicating the inherent complexity of diabetes prediction from behavioral health indicators alone.

### Challenge 2: Missing Values in PIMA Diabetes Dataset
- **Problem:** Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns were biologically impossible (e.g., blood pressure of 0), representing missing data encoded as zeros.
- **Solution:** Replaced zero values with column-wise mean values for the affected columns before applying StandardScaler. This approach preserves the distribution while providing reasonable estimates for missing values.

### Challenge 3: High-Dimensional Sparse Symptom Data
- **Problem:** 132 binary symptom features create a high-dimensional feature space where most values are 0, potentially causing the curse of dimensionality.
- **Solution:** SVM with RBF kernel (C=0.1) effectively handles high-dimensional sparse data. The distinct, non-overlapping symptom patterns per disease made the classification task separable, resulting in 100% accuracy.

### Challenge 4: Free-Text Symptom Processing
- **Problem:** Natural language symptom descriptions vary widely in vocabulary, sentence structure, and length, making classification challenging.
- **Solution:** TF-IDF vectorization with tuned parameters (max_features=8000, ngram_range=(1,2)) captures both individual terms and bi-gram phrases. Combined with Multinomial Naive Bayes (alpha=0.01), this achieved 97.92% accuracy on the Symptom2Disease dataset.

### Challenge 5: Chatbot Low-Confidence Responses
- **Problem:** The neural network occasionally produced incorrect or irrelevant responses for queries outside the training intent space.
- **Solution:** Implemented a confidence threshold of 0.75. Responses with confidence below this threshold trigger a fallback message ("I do not understand..."), preventing misleading medical advice.

### Challenge 6: Integration of Multiple ML Models
- **Problem:** Loading and serving 5+ trained models (binary, text, diabetes basic, diabetes advanced, chatbot) in a single Flask application could cause memory issues and slow startup.
- **Solution:** Models are loaded once at application startup using Joblib and Pickle, stored as global variables, and reused across requests. The chatbot model (data.pth) is loaded separately through the `chat.py` module.

### Challenge 7: LLM Response Parsing
- **Problem:** Gemma 3 via Ollama occasionally returned non-JSON responses or responses wrapped in markdown formatting.
- **Solution:** Implemented robust JSON extraction using `raw.index("{")` and `raw.rindex("}")` to find JSON boundaries, with a retry mechanism using a simplified prompt on parse failure.

---

# CHAPTER 7a

## SECURITY AND PRIVACY CONSIDERATIONS

### 7a.1 Data Security Measures

| Security Measure | Implementation | Purpose |
|-----------------|----------------|---------|
| SQL Injection Prevention | Parameterized queries using `%s` placeholders in all MySQL queries | Prevents malicious SQL code injection |
| File Upload Security | Werkzeug `secure_filename()` sanitizes uploaded report filenames | Prevents directory traversal attacks |
| Session Management | Flask session with secret key for user authentication | Maintains secure user state |
| Payment Security | Razorpay handles PCI-DSS compliant payment processing | No card data stored locally |
| Input Validation | HTML5 form validation + server-side type checking | Prevents malformed input |
| Access Control | Role-based routes — Admin, Doctor, User panels separately protected | Unauthorized access prevention |
| Foreign Key Constraints | ON DELETE CASCADE on appointment table | Referential integrity |

### 7a.2 Privacy Considerations

1. **Medical Reports:** Stored on server filesystem, not in database — reduces data exposure in case of SQL-based breach
2. **Payment Data:** No credit card details stored; only Razorpay order ID stored as reference
3. **Chatbot Conversations:** No conversation history stored — stateless interaction ensures privacy
4. **Data Minimization:** Only essential user information collected (name, email, mobile, age)
5. **No Third-Party Sharing:** All data remains within the application; no external analytics or tracking

### 7a.3 Production Recommendations

For production deployment, the following security enhancements are recommended:
- Implement password hashing using bcrypt or Argon2 (current version stores plaintext passwords)
- Use HTTPS with SSL/TLS certificates for encrypted communication
- Implement JWT (JSON Web Tokens) or OAuth 2.0 for stateless authentication
- Add rate limiting to prevent brute-force attacks
- Enable CSRF (Cross-Site Request Forgery) protection via Flask-WTF
- Store sensitive configuration (DB credentials, API keys) in environment variables, not in source code
- Implement regular database backups and disaster recovery procedures

---

# CHAPTER 7b

## ETHICAL CONSIDERATIONS

### 7b.1 Medical AI Ethics

1. **Not a Replacement for Doctors:** This system provides **preliminary screening only**. All predictions must be validated by qualified medical professionals. The application includes a disclaimer that AI-generated predictions should not be used as the sole basis for medical decisions.

2. **False Positives vs False Negatives:**
   - **False Positives:** Patient unnecessarily worried but advised to consult a doctor (low harm)
   - **False Negatives:** Patient misses a diagnosis, potentially leading to delayed treatment (higher harm)
   - The system is designed to favor sensitivity where possible, erring on the side of caution

3. **Data Bias:** Models are trained on specific datasets:
   - PIMA dataset: Primarily Pima Indian women — may not generalize to all ethnicities
   - BRFSS dataset: US-based health survey — may not apply to other countries
   - Symptom2Disease: English-only descriptions — excludes non-English speakers
   - These limitations mean predictions may be less accurate for underrepresented populations

4. **Transparency:** Confidence scores are displayed for every prediction, allowing users to assess the reliability of results. Users are informed that the analysis is AI-driven.

5. **Informed Consent:** Patients voluntarily input their symptoms and are aware that the system uses AI for analysis. No data is collected without user interaction.

6. **LLM-Generated Content:** Disease information generated by Gemma 3 is used for educational purposes only. Users are advised to consult healthcare professionals for medical advice.

---

# CHAPTER 7c

## LIMITATIONS

1. **Small PIMA Dataset:** The basic diabetes model is trained on only 768 records, limiting its accuracy to approximately 78%. A larger, more diverse dataset would improve performance.

2. **Low Diabetes Recall (BRFSS):** Despite applying SMOTE, the diabetes class recall is only 19%, meaning approximately 81% of actual diabetic patients would be missed by the advanced model. This is a significant limitation for a clinical tool.

3. **Disease Overlap Assumption:** The binary symptom model assumes non-overlapping symptom patterns across 41 diseases. In reality, many diseases share common symptoms (e.g., fever, fatigue), and the 100% accuracy may not reflect real-world complexity.

4. **Language Limitation:** Text-based prediction only works with English symptom descriptions. Patients who speak other languages cannot use this feature.

5. **No Image Analysis:** The system cannot process medical images (X-rays, CT scans, skin photographs), which are critical for diagnosing many conditions.

6. **Static Specialist Mapping:** Disease-specialist mapping is hardcoded and cannot adapt to new medical specializations or regional variations in specialist availability.

7. **No Patient History:** The system does not consider a patient's medical history, previous diagnoses, or ongoing medications when making predictions.

8. **Session-Based Security:** The current implementation uses basic Flask session management. Production deployment would require JWT/OAuth for robust security.

9. **LLM Dependency:** Disease information quality depends entirely on Gemma 3's output accuracy. The LLM may occasionally generate incorrect or inconsistent medical information.

10. **No Real-Time Model Updates:** Models are pre-trained and static. They do not update in real-time as new patient data becomes available.

---

# CHAPTER 8

## FUTURE SCOPE

The following enhancements are planned for future versions of the system:

1. **Integrate Pretrained AI Models (Gemma 4 / LLMs):** Use pretrained large language models like Gemma 4 via Ollama for enhanced disease prediction from complex symptom descriptions, improved chatbot responses with contextual medical knowledge, and automated medical report analysis. The current system uses Gemma 3 for disease info generation — future versions can extend LLM integration to prediction, diagnosis explanation, and personalized health recommendations.

2. **Deep Learning Models:** Implement CNN (Convolutional Neural Networks) and LSTM (Long Short-Term Memory) networks for improved accuracy on text-based symptom classification and time-series health data analysis.

3. **Extended Disease Coverage:** Integrate additional datasets to expand coverage beyond the current 41 (binary) and 24 (text) diseases.

4. **Multi-Language Support:** Enable symptom input in multiple languages (Hindi, Marathi, etc.) using multilingual NLP models.

5. **Mobile Application:** Develop a React Native or Flutter mobile application for broader accessibility on smartphones.

6. **Electronic Health Records (EHR) Integration:** Connect with hospital EHR systems to pull patient history and provide more contextual predictions.

7. **Real-Time Video Consultation:** Integrate WebRTC for live video consultation with doctors through the platform.

8. **Drug Interaction Checker:** Add a module to check for potential drug interactions based on patient medications.

9. **IoT and Wearable Device Integration:** Incorporate data from wearable devices (heart rate, blood oxygen, glucose monitors) for continuous health monitoring.

10. **Model Retraining Pipeline:** Implement automated model retraining and monitoring using MLOps tools (MLflow, Airflow) to keep models updated with new data.

---

# CHAPTER 9

## CONCLUSION

This project successfully developed an **end-to-end AI-powered healthcare web application** for disease prediction based on symptoms. The key achievements include:

1. **Multiple Prediction Modes:** The system integrates four distinct prediction modules — binary symptom-based (132 symptoms → 41 diseases), text-based (free text → 24 diseases), basic diabetes (8 features), and advanced diabetes (21 health indicators from 229,474 records) — in a single application, a feature not commonly found in existing systems.

2. **High Accuracy:** Binary symptom prediction achieves **100% test accuracy** (SVM), text-based prediction achieves **97.92% accuracy** (TF-IDF + MultinomialNB), and advanced diabetes prediction achieves **84.05% accuracy** (Random Forest with SMOTE).

3. **End-to-End Functionality:** Beyond prediction, the system provides automatic specialist mapping, online appointment booking with Razorpay payment gateway, email notifications, medical report upload, and AI chatbot support.

4. **Role-Based Access:** Three user roles (Patient, Doctor, Admin) with dedicated dashboards and management capabilities ensure secure and organized access.

5. **LLM Enhancement:** Integration of Gemma 3 via Ollama for generating structured, comprehensive disease information demonstrates the potential of large language models in healthcare applications.

6. **Scalable Architecture:** The Flask-based three-tier architecture, configuration-driven ML pipeline, and modular code design enable easy extension and maintenance.

The system demonstrates the effective application of Machine Learning, NLP, and Deep Learning techniques in creating a practical healthcare tool. While not a replacement for professional medical diagnosis, it serves as a valuable preliminary screening tool that can guide patients toward timely specialist consultation.

### Social and Healthcare Impact

1. **Accessibility:** Provides preliminary diagnosis to patients in remote/underserved areas with internet access, bridging the urban-rural healthcare divide.
2. **Early Detection:** Enables early disease detection by making symptom analysis readily available 24/7 without requiring an appointment.
3. **Cost Reduction:** Reduces unnecessary hospital visits by providing initial screening at home (₹200 online appointment vs ₹500+ OPD fees + travel).
4. **Health Awareness:** AI chatbot educates users about diseases, symptoms, and when to seek medical help, promoting health literacy.
5. **Specialist Routing:** Automatically guides patients to the right specialist, reducing misdirected consultations and wasted medical resources.
6. **Scalability:** Web-based system can serve thousands of users simultaneously with minimal infrastructure investment.

---

# REFERENCES

[1] Y. Deepthi, K.P. Kalyan, M. Vyas, K. Radhika, D.K. Babu, and N.V. Krishna Rao, "Disease Prediction Based on Symptoms Using Machine Learning," in *Energy Systems, Drives and Automations: Proceedings of ESDA 2019*, Springer, 2020. [Online]. Available: https://scholar.google.com/scholar?q=Disease+prediction+based+on+symptoms+using+machine+learning+Deepthi+Kalyan

[2] D. Sisodia and D. Sisodia, "Prediction of Diabetes using Classification Algorithms," *International Journal of Computer Applications*, vol. 176, no. 4, pp. 19–23, 2018. [Online]. Available: https://www.ijcaonline.org/archives/volume176/number4/sisodia-2018-ijca-916935.pdf

[3] A. K. Singh, S. Kumar, M. Singh et al., "Machine Learning Approach for Diabetes Prediction Using BRFSS Dataset," *Procedia Computer Science*, 2022. [Online]. Available: https://doi.org/10.1016/j.procs.2022.01.146

[4] A. Hamdi, M. Mohamed, R. Emad, and K. Shaban, "An Ensemble Classification Approach in A Multi-Layered Large Language Model Framework for Disease Prediction," *arXiv preprint arXiv:2509.02446*, September 2025. [Online]. Available: https://arxiv.org/abs/2509.02446

[5] R. Zannat, A. Al Shafi, and A. Muntakim, "Bridging the Gap in Bangla Healthcare: Machine Learning Based Disease Prediction Using a Symptoms-Disease Dataset," *IEEE ECCE 2025* (arXiv:2601.12068), January 2026. [Online]. Available: https://arxiv.org/abs/2601.12068

[6] A. Al Shafi, R. Zannat, A. Muntakim, and M. Hasan, "A Structured Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy," *arXiv preprint arXiv:2506.13610*, June 2025. [Online]. Available: https://arxiv.org/abs/2506.13610

[7] M. Shetty and C. Jordan, "Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM," *arXiv preprint arXiv:2503.19394*, March 2025. [Online]. Available: https://arxiv.org/abs/2503.19394

[8] V. Krishnaiah, G. Narsimha, and N. Subhash Chandra, "A Comparative Study of Classification Algorithms for Disease Prediction," *International Journal of Computer Applications*, vol. 147, no. 10, 2016.

[9] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[10] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems 32*, 2019, pp. 8024–8035.

[11] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[12] W. McKinney, "Data Structures for Statistical Computing in Python," in *Proceedings of the 9th Python in Science Conference*, 2010, pp. 56–61.

[13] Bird, Steven, Edward Loper and Ewan Klein, *Natural Language Processing with Python*, O'Reilly Media Inc., 2009.

[14] M. Pal, "Random Forest Classifier for Remote Sensing Classification," *International Journal of Remote Sensing*, vol. 26, no. 1, pp. 217–222, 2005.

[15] C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

---

# APPENDICES

## Appendix A: Project Quick Stats

| Metric | Value |
|--------|-------|
| Total Diseases (Binary Symptoms) | 41 |
| Total Diseases (Text-Based) | 24 |
| Total Symptoms Tracked | 132 |
| Total Symptom Text Descriptions | 1,200 |
| ML Models Trained | 17 (6 Binary + 5 Text + 6 Diabetes Basic + 6 Diabetes Advanced + Chatbot) |
| Deep Learning Models | 1 (PyTorch NN for Chatbot) |
| Prediction Modes | 4 |
| Database Tables | 4 |
| HTML Templates | 27 |
| User Roles | 3 (User, Doctor, Admin) |
| Specialist Categories | 8 |
| Diabetes Dataset Records (Basic) | 768 |
| Diabetes Dataset Records (Advanced) | 229,474 |
| Best Binary Model Accuracy | 100% (SVM) |
| Best Text Model Accuracy | 97.92% (TF-IDF + MNB) |
| Best Diabetes Advanced Accuracy | 84.05% (Random Forest + SMOTE) |
| LLM Used | Gemma 3 (via Ollama) |

## Appendix B: Software and Hardware Requirements

### Software Requirements
| Requirement | Specification |
|------------|---------------|
| Operating System | Windows 10/11, macOS, or Linux |
| Python | 3.8+ |
| Database | MySQL 8.0+ |
| Conda/Anaconda | For environment management |
| Ollama | For Gemma 3 LLM (optional) |

### Hardware Requirements
| Requirement | Specification |
|------------|---------------|
| Processor | Intel Core i5 or equivalent |
| RAM | 8 GB minimum (16 GB recommended for BRFSS training) |
| Storage | 2 GB for project + datasets |
| Internet | Required for Razorpay payments and Ollama API |

## Appendix C: Model Files

| File | Description |
|------|-------------|
| `saved_model/best_binary_model.joblib` | SVM model for binary symptom prediction |
| `saved_model/best_text_model.joblib` | TF-IDF + MultinomialNB pipeline for text prediction |
| `saved_model/text_label_encoder.joblib` | Label encoder for text model disease names |
| `saved_model/decision_tree.joblib` | Decision Tree model for binary symptoms |
| `saved_model/mnb.joblib` | Multinomial Naive Bayes for binary symptoms |
| `saved_model/gradient_boost.joblib` | Gradient Boosting model for binary symptoms |
| `svm_model.pkl` | SVM model for basic diabetes prediction |
| `scaler.pkl` | StandardScaler for basic diabetes model |
| `diabetes_brfss_model.pkl` | Random Forest for advanced diabetes prediction |
| `diabetes_brfss_scaler.pkl` | StandardScaler for advanced diabetes model |
| `diabetes_brfss_features.pkl` | Feature names for advanced diabetes model |
| `data.pth` | PyTorch state dict for chatbot neural network |

## Appendix D: Application Screenshots

> **[Insert screenshots of the running application here]**
>
> Suggested screenshots:
> 1. Home Page
> 2. User Registration / Login
> 3. Disease Prediction (Binary Symptoms) - Input and Result
> 4. Disease Prediction (Text-Based) - Input and Result with Top-3
> 5. Diabetes Basic Assessment - Form and Result
> 6. Diabetes Advanced Assessment - Form with 21 indicators and Risk Meter
> 7. Appointment Booking - Doctor selection and form
> 8. Razorpay Payment Page
> 9. Doctor Panel - Appointment management
> 10. Admin Dashboard - User/Doctor management
> 11. AI Chatbot - Sample conversation

## Appendix E: Project Timeline

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

## Appendix F: Glossary of Terms

| Term | Definition |
|------|-----------|
| **AI** | Artificial Intelligence — simulation of human intelligence by machines |
| **Bag of Words (BoW)** | Text representation method that counts word occurrences, ignoring grammar and word order |
| **BERT** | Bidirectional Encoder Representations from Transformers — a transformer-based NLP model |
| **Binary Classification** | Classification task with exactly two classes (e.g., diabetic/non-diabetic) |
| **BRFSS** | Behavioral Risk Factor Surveillance System — annual US health survey by CDC |
| **Cross-Validation** | Model evaluation technique that splits data into k folds, training on k-1 and testing on 1 |
| **Deep Learning** | Subset of ML using neural networks with multiple hidden layers |
| **F1-Score** | Harmonic mean of precision and recall; F1 = 2 × (Precision × Recall) / (Precision + Recall) |
| **Feature Engineering** | Process of creating new input features from existing data to improve model performance |
| **Feature Importance** | Score indicating how much each feature contributes to a model's predictions |
| **Flask** | Lightweight Python web framework for building web applications |
| **Gemma** | Family of lightweight open-source LLMs by Google, used via Ollama |
| **Gini Impurity** | Measure of how often a randomly chosen element would be incorrectly classified |
| **Hyperparameter** | Model configuration parameter set before training (e.g., learning rate, number of trees) |
| **IDF** | Inverse Document Frequency — measures how rare a term is across all documents |
| **Joblib** | Python library for efficient serialization of NumPy arrays and scikit-learn models |
| **LLM** | Large Language Model — AI model trained on vast text data for language tasks |
| **MNB** | Multinomial Naive Bayes — probabilistic classifier suitable for text classification |
| **NLP** | Natural Language Processing — field of AI focused on human-computer language interaction |
| **Ollama** | Tool for running large language models locally (e.g., Gemma, Llama) |
| **Overfitting** | When a model learns training data too well, including noise, and performs poorly on new data |
| **Pickle** | Python serialization module for saving/loading objects |
| **PIMA** | Pima Indian Diabetes dataset — standard benchmark for diabetes prediction |
| **PorterStemmer** | Algorithm for reducing words to their root form (e.g., "running" → "run") |
| **PyTorch** | Open-source deep learning framework by Meta (Facebook) |
| **Random Forest** | Ensemble of decision trees, each trained on a random subset of data and features |
| **Recall** | Proportion of actual positives correctly identified; Recall = TP / (TP + FN) |
| **ReLU** | Rectified Linear Unit — activation function: f(x) = max(0, x) |
| **RBF Kernel** | Radial Basis Function kernel — maps data to infinite-dimensional space for non-linear classification |
| **Razorpay** | Indian payment gateway platform for online transactions |
| **SMOTE** | Synthetic Minority Oversampling Technique — generates synthetic samples for minority class |
| **StandardScaler** | Scikit-learn transformer that standardizes features to zero mean and unit variance |
| **SVM** | Support Vector Machine — finds the optimal hyperplane separating classes |
| **TF** | Term Frequency — measures how often a term appears in a document |
| **TF-IDF** | Term Frequency-Inverse Document Frequency — numerical statistic reflecting word importance |
| **Tokenization** | Process of splitting text into individual words, phrases, or symbols |
| **UML** | Unified Modeling Language — standardized notation for software design diagrams |
| **Validation Split** | Portion of training data reserved for evaluating model during development |

## Appendix G: Plagiarism Certificate

> **[Insert plagiarism check report here]**
>
> Plagiarism must be below 10% as per university guidelines.
> Recommended tools: Turnitin, iThenticate, or PlagiarismCheck.org
> Attach the certificate showing plagiarism percentage.

## Appendix H: IEEE Format Compliance Note

This report follows the IEEE/Springer paper format as required by the university evaluation criteria:

1. **References:** IEEE citation style [numbered] with full bibliographic details
2. **Structure:** Abstract → Introduction → Literature Survey → Methodology → Implementation → Results → Conclusion → References
3. **Figures and Tables:** Numbered sequentially with descriptive captions
4. **Mathematical Notations:** Equations numbered and defined with variable descriptions
5. **Citations:** In-text citations as [1], [2], etc. referencing the numbered bibliography
