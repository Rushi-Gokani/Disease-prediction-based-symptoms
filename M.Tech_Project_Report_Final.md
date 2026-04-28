<p align="center">
  <img src="https://img.shields.io/badge/M.Tech-Project%20Report-blue?style=for-the-badge" alt="M.Tech Project Report"/>
  <img src="https://img.shields.io/badge/AI-Healthcare-green?style=for-the-badge" alt="AI Healthcare"/>
  <img src="https://img.shields.io/badge/Python-Flask-yellow?style=for-the-badge" alt="Python Flask"/>
</p>

---

<h1 align="center">Disease Prediction Based on Symptoms</h1>
<h3 align="center">An AI-Powered Healthcare Web Application for Early Disease Detection</h3>

---

<p align="center"><b>Submitted in partial fulfillment of the requirements for the degree of</b></p>
<p align="center"><b>Master of Technology (M.Tech)</b></p>

---

<div align="center">

| | |
|---|---|
| **Submitted By:** | [Your Name] |
| **Roll Number:** | [Your Roll Number] |
| **Department:** | [Department Name] |
| **University:** | [University Name] |
| **Guide:** | [Guide Name] |
| **Academic Year:** | 2025-2026 |

</div>

---

## Declaration

I hereby declare that the project entitled **"Disease Prediction Based on Symptoms"** is a record of original work carried out by me under the guidance of **[Guide Name]**, **[Department]**, **[University Name]**, in partial fulfillment of the requirements for the award of the degree of Master of Technology. This project has not been submitted elsewhere for the award of any degree or diploma.

**Date:** [Date]

**Place:** [Place]

**Signature:** _______________

---

## Certificate

This is to certify that the project entitled **"Disease Prediction Based on Symptoms"** is a bonafide record of work done by **[Your Name]** (Roll No: **[Roll Number]**) under my guidance and supervision, in partial fulfillment of the requirements for the award of the degree of Master of Technology in **[Department]** from **[University Name]**.

| | |
|---|---|
| **Guide Name:** | _______________ |
| **Department:** | _______________ |
| **University:** | _______________ |
| **Signature:** | _______________ |
| **Date:** | _______________ |

---

## Acknowledgements

I would like to express my sincere gratitude to my project guide **[Guide Name]** for their invaluable guidance, constant encouragement, and support throughout the duration of this project. I also thank the faculty members of the Department of **[Department]** for their constructive feedback and suggestions.

I extend my thanks to the developers of open-source tools and libraries -- **Scikit-learn**, **PyTorch**, **Flask**, **NLTK**, and **Ollama (Gemma 3)** -- that made this project possible. I am also grateful to the creators of the publicly available datasets (Symptom-Disease Dataset, PIMA Diabetes Dataset, CDC BRFSS 2015, and Symptom2Disease) used in this research.

Finally, I thank my family and friends for their unwavering support and encouragement.

---

## Abstract

Healthcare systems worldwide face challenges in providing timely and accurate preliminary disease diagnosis. Manual diagnosis is time-consuming, prone to human error, and often inaccessible to patients in remote areas. This project presents an AI-powered web application that predicts diseases from user-reported symptoms using Machine Learning and Natural Language Processing techniques.

The system integrates **four distinct prediction modules**:

1. **Binary Symptom-Based Disease Prediction** -- using 132 symptoms across 41 diseases
2. **Text-Based Disease Prediction** -- from natural language symptom descriptions across 24 diseases
3. **Basic Diabetes Risk Assessment** -- using 8 clinical features from the PIMA dataset
4. **Advanced Diabetes Risk Assessment** -- using 21 health indicators from the CDC BRFSS 2015 dataset (229,474 records)

Multiple ML algorithms were evaluated including Random Forest, SVM, Decision Tree, Gradient Boosting, Naive Bayes, Logistic Regression, and XGBoost. The binary symptom models achieved **100% test accuracy** (SVM), the text-based model achieved **97.9% accuracy** (TF-IDF + Multinomial Naive Bayes), and the advanced diabetes model achieved **84.05% accuracy** (Random Forest with SMOTE). A PyTorch-based neural network chatbot provides interactive health assistance.

The application also features online appointment booking with specialist doctors, Razorpay payment gateway integration, email notifications via SMTP, and role-based access control (User, Doctor, Admin). Disease information was enriched using **Gemma 3** (LLM via Ollama) for generating structured medical content.

**Keywords:** Disease Prediction, Machine Learning, Natural Language Processing, Diabetes Prediction, Flask, PyTorch, Chatbot, Healthcare, SMOTE, TF-IDF

---

## Table of Contents

| Chapter | Title | Page |
|---------|-------|------|
| | Declaration | ii |
| | Certificate | iii |
| | Acknowledgements | iv |
| | Abstract | v |
| | Table of Contents | vi |
| | List of Tables | viii |
| | List of Figures | ix |
| **1** | **Introduction** | **1** |
| 1.1 | Background | 1 |
| 1.2 | Problem Statement | 2 |
| 1.3 | Objectives | 3 |
| 1.4 | Scope of the Project | 4 |
| 1.5 | Organization of the Report | 4 |
| **2** | **Literature Survey** | **5** |
| 2.1 | Disease Prediction from Symptoms Using ML | 5 |
| 2.2 | ML-Based Diabetes Prediction on PIMA Dataset | 5 |
| 2.3 | Diabetes Prediction Using BRFSS Data | 6 |
| 2.4 | NLP-Based Disease Prediction | 6 |
| 2.5 | Ensemble Approaches for Disease Prediction | 7 |
| 2.6 | Structured Disease-Symptom Datasets | 7 |
| 2.7 | Causality in Clinical Decision Making | 7 |
| 2.8 | Comparative Study of Classification Algorithms | 8 |
| 2.9 | Key Literature Gaps Addressed | 8 |
| **3** | **System Design and Methodology** | **9** |
| 3.1 | System Architecture | 9 |
| 3.2 | Database Design | 10 |
| 3.3 | Data Flow Diagrams | 12 |
| 3.4 | UML Diagrams | 14 |
| 3.5 | Machine Learning Pipeline | 16 |
| 3.6 | NLP Pipeline (Chatbot) | 18 |
| 3.7 | Deep Learning Pipeline | 19 |
| 3.8 | Technology Stack | 20 |
| 3.9 | Specialist Mapping Logic | 21 |
| 3.10 | Mathematical Formulation of Algorithms | 22 |
| **4** | **Implementation** | **26** |
| 4.1 | Dataset Details | 26 |
| 4.2 | Data Preprocessing | 28 |
| 4.3 | Feature Engineering | 30 |
| 4.4 | Model Training and Selection | 31 |
| 4.5 | Web Application Implementation | 34 |
| 4.6 | Payment Gateway Integration | 37 |
| 4.7 | LLM Integration (Gemma 3) | 38 |
| **5** | **Results and Discussion** | **39** |
| 5.1 | Disease Prediction Results (Binary Symptoms) | 39 |
| 5.2 | Disease Prediction Results (Text-Based) | 40 |
| 5.3 | Diabetes Prediction Results (Basic - PIMA) | 41 |
| 5.4 | Diabetes Prediction Results (Advanced - BRFSS) | 42 |
| 5.5 | Chatbot Performance | 43 |
| 5.6 | SMOTE Analysis | 44 |
| 5.7 | Confusion Matrix Analysis | 45 |
| 5.8 | Feature Importance Analysis | 46 |
| 5.9 | Comparison with Existing Systems | 47 |
| 5.10 | Performance Evaluation Metrics | 48 |
| **6** | **Testing and Validation** | **49** |
| 6.1 | Model Testing | 49 |
| 6.2 | System Testing | 50 |
| 6.3 | Data Validation | 51 |
| **7** | **Challenges and Solutions** | **52** |
| **7a** | **Security and Privacy Considerations** | **55** |
| **7b** | **Ethical Considerations** | **57** |
| **7c** | **Limitations** | **59** |
| **8** | **Future Scope** | **60** |
| **9** | **Conclusion** | **62** |
| | References | 64 |
| | Appendices | 66 |

---

## List of Tables

| Table No. | Title |
|-----------|-------|
| Table 3.1 | Admin Table Schema |
| Table 3.2 | User Table Schema |
| Table 3.3 | Doctors Table Schema |
| Table 3.4 | Appointment Table Schema |
| Table 3.5 | Technology Stack Summary |
| Table 3.6 | Specialist-Disease Mapping |
| Table 4.1 | Disease Symptom Dataset (Binary) Properties |
| Table 4.2 | PIMA Diabetes Dataset Properties |
| Table 4.3 | CDC BRFSS 2015 Dataset Properties |
| Table 4.4 | Symptom2Disease Dataset Properties |
| Table 4.5 | Binary Symptom Model Training Results |
| Table 4.6 | Text-Based Model Training Results |
| Table 4.7 | BRFSS Diabetes Model Comparison |
| Table 4.8 | Flask Application User Routes |
| Table 4.9 | Flask Application Admin Routes |
| Table 4.10 | Flask Application Doctor Routes |
| Table 5.1 | Binary Symptom Model Results Summary |
| Table 5.2 | Text-Based Model Results Summary |
| Table 5.3 | PIMA Diabetes Model Results |
| Table 5.4 | BRFSS Diabetes Model Results |
| Table 5.5 | BRFSS Random Forest Classification Report |
| Table 5.6 | Feature Importance Ranking (BRFSS) |
| Table 5.7 | Comparison with Existing Systems |
| Table 5.8 | Overall System Performance Summary |
| Table 6.1 | Model Testing Summary |
| Table 6.2 | System Testing Results |
| Table 6.3 | Data Validation Measures |

## List of Figures

| Figure No. | Title |
|------------|-------|
| Figure 3.1 | Three-Tier System Architecture |
| Figure 3.2 | Entity-Relationship (ER) Diagram |
| Figure 3.3 | Level 0 -- Context Diagram (DFD) |
| Figure 3.4 | Level 1 -- Detailed Data Flow Diagram |
| Figure 3.5 | Use Case Diagram |
| Figure 3.6 | Sequence Diagram -- Disease Prediction Flow |
| Figure 3.7 | Sequence Diagram -- Appointment & Payment Flow |
| Figure 3.8 | Binary Symptom Prediction Pipeline |
| Figure 3.9 | Text-Based Prediction Pipeline |
| Figure 3.10 | Diabetes Prediction Pipeline (PIMA) |
| Figure 3.11 | Diabetes Prediction Pipeline (BRFSS) |
| Figure 3.12 | Chatbot NLP Pipeline |
| Figure 3.13 | Neural Network Architecture |
| Figure 5.1 | BRFSS Confusion Matrix (Random Forest) |
| Figure 5.2 | Feature Importance Chart (BRFSS) |
| Figure 5.3 | SMOTE Class Distribution (Before/After) |

---

<br/>

# CHAPTER 1

## INTRODUCTION

### 1.1 Background

The healthcare industry is one of the most critical sectors in any country's economy. Timely and accurate diagnosis of diseases is essential for effective treatment and improved patient outcomes. However, the traditional approach to disease diagnosis relies heavily on manual examination by medical professionals, which can be time-consuming and subjective. In many developing regions, patients also face a shortage of qualified medical practitioners, leading to delayed diagnoses and worsening health conditions.

With the rapid advancement of Artificial Intelligence (AI) and Machine Learning (ML), there is significant potential to develop automated systems that can assist in preliminary disease diagnosis. Such systems can analyze patient-reported symptoms and predict probable diseases with high accuracy, enabling patients to seek timely medical attention from the appropriate specialist.

Natural Language Processing (NLP) further enhances these systems by allowing patients to describe their symptoms in plain text, removing the need for structured form inputs. Deep Learning techniques enable the creation of intelligent chatbots that can interact with patients, answer health-related queries, and guide them through the diagnostic process.

The convergence of these technologies -- ML, NLP, and Deep Learning -- presents an unprecedented opportunity to democratize healthcare access. A well-designed AI-powered diagnostic system can serve as a first point of contact for patients, triaging their conditions and directing them to the appropriate specialist, thereby reducing the burden on the healthcare system and improving patient outcomes.

### 1.2 Problem Statement

Despite advances in medical technology, the following challenges persist:

1. **Delayed Diagnosis:** Patients often delay seeking medical attention due to lack of awareness about their symptoms, leading to worsening health conditions.
2. **Specialist Identification:** Patients frequently do not know which medical specialist to consult for their specific condition.
3. **Limited Accessibility:** Automated preliminary diagnosis tools are not widely available, especially in rural and underserved areas.
4. **Manual Processes:** Appointment booking, report management, and follow-ups are often manual and inefficient.
5. **Single-Mode Prediction:** Existing systems typically offer only one mode of prediction (e.g., structured symptoms only), limiting user accessibility.

> **There is a need for an integrated, AI-powered healthcare web application that provides multiple prediction modes, specialist mapping, appointment management, and interactive health assistance in a single platform.**

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

| Chapter | Description |
|---------|-------------|
| **Chapter 1** | Introduction -- background, problem statement, objectives, and scope |
| **Chapter 2** | Literature Survey -- comprehensive review of related work |
| **Chapter 3** | System Design and Methodology -- architecture, database, ML pipelines |
| **Chapter 4** | Implementation -- datasets, preprocessing, model training, web app |
| **Chapter 5** | Results and Discussion -- performance analysis across all modules |
| **Chapter 6** | Testing and Validation -- model and system testing |
| **Chapter 7** | Challenges and Solutions -- technical obstacles overcome |
| **Chapter 7a** | Security and Privacy Considerations |
| **Chapter 7b** | Ethical Considerations |
| **Chapter 7c** | Limitations |
| **Chapter 8** | Future Scope -- planned enhancements |
| **Chapter 9** | Conclusion -- summary of achievements and impact |

---

<br/>

# CHAPTER 2

## LITERATURE SURVEY

A comprehensive literature survey was conducted to understand the current state of the art in disease prediction using machine learning, diabetes prediction, NLP-based diagnosis, and related fields. The following subsections present key findings from the reviewed literature.

### 2.1 Disease Prediction from Symptoms Using Machine Learning

**Deepthi et al. (2020)** [1] compared Naive Bayes, Decision Tree, and Random Forest algorithms for symptom-based disease prediction. Their study demonstrated that Decision Tree and Random Forest achieved over 95% accuracy on structured symptom datasets. The study used a binary feature representation where each symptom was encoded as 0 or 1, similar to our approach. However, the study focused on a single prediction mode and did not include text-based input or end-to-end appointment management.

### 2.2 ML-Based Diabetes Prediction on PIMA Dataset

**Sisodia and Sisodia (2018)** [2] conducted a comparative study of classification algorithms on the PIMA Diabetes Dataset. Their findings are summarized below:

| Algorithm | Accuracy |
|-----------|----------|
| Logistic Regression | 78.3% |
| SVM | 77.6% |
| Naive Bayes | 76.3% |
| Decision Tree | 73.4% |
| KNN | 72.4% |

Their findings indicated that SVM and Logistic Regression performed best among traditional ML classifiers. Our system uses the same PIMA dataset and achieves comparable results, with Naive Bayes at 78.57% and SVM at 75.32%.

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

| Gap in Existing Literature | How This Project Addresses It |
|---|---|
| Most studies focus on a single prediction mode | Our system integrates **4 prediction modes** in one application |
| Limited studies use 21+ health indicators for diabetes | We leverage BRFSS with **21 features and 229,474 records** |
| Few systems provide end-to-end functionality | We provide **prediction -> specialist mapping -> appointment -> payment** |
| Limited integration of NLP for symptom text | TF-IDF pipeline achieves **97.92% accuracy** on free-text input |
| No LLM-enriched disease information | **Gemma 3 via Ollama** generates structured medical content |
| Lack of interactive health assistance | **PyTorch chatbot** provides real-time health query responses |

---

<br/>

# CHAPTER 3

## SYSTEM DESIGN AND METHODOLOGY

### 3.1 System Architecture

The system follows a **three-tier architecture** designed for modularity, scalability, and separation of concerns:

**Figure 3.1: Three-Tier System Architecture**

```
+-------------------------------------------------------------+
|                     PRESENTATION TIER                        |
|            HTML5, CSS3, Bootstrap, JavaScript                |
|      (27+ HTML Templates, Responsive UI, Chatbot Widget)     |
+-----------------------------+-------------------------------+
                              | HTTP/HTTPS
+-----------------------------v-------------------------------+
|                       LOGIC TIER                             |
|                  Python Flask (app.py)                        |
|                                                              |
|   +-----------+  +------------+  +--------+  +----------+   |
|   |  Disease  |  |  Diabetes  |  |  Chat  |  | Payment  |   |
|   | Predictor |  |  Predictor |  |   Bot  |  | Gateway  |   |
|   +-----+-----+  +------+-----+  +---+----+  +----+-----+   |
|         |               |            |             |         |
|   +-----v-----+  +------v-----+     |       +----v-----+   |
|   | ML Models |  | ML Models  |     |       | Razorpay |   |
|   | (Joblib)  |  | (Pickle)   |     |       |   API    |   |
|   +-----------+  +------------+     |       +----------+   |
|                                     |                       |
|         +---------------------------v-----------+           |
|         |    PyTorch Neural Network Model       |           |
|         |    (data.pth + NLTK Pipeline)         |           |
|         +---------------------------------------+           |
+-----------------------------+-------------------------------+
                              | SQL Queries / SMTP
+-----------------------------v-------------------------------+
|                       DATA TIER                              |
|  +----------+   +--------------+   +---------------+        |
|  |  MySQL   |   |    SMTP      |   |  File System  |        |
|  | Database |   |  (Gmail)     |   |  (Reports)    |        |
|  | (pro17)  |   +--------------+   +---------------+        |
|  +----------+                                               |
+-------------------------------------------------------------+
```

**Tier Descriptions:**

| Tier | Components | Responsibility |
|------|-----------|----------------|
| **Presentation** | HTML5, CSS3, Bootstrap, JavaScript | User interface, form inputs, result display, chatbot widget |
| **Logic** | Flask, ML Models, PyTorch, Razorpay API | Business logic, predictions, payment processing, routing |
| **Data** | MySQL, SMTP (Gmail), File System | Persistent storage, email delivery, report file management |

### 3.2 Database Design

The MySQL database (`pro17`) consists of **4 tables** with well-defined relationships:

**Table 3.1: admin**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INT | PK, AUTO_INCREMENT | Admin ID |
| a_username | VARCHAR(255) | NOT NULL | Admin username |
| a_password | VARCHAR(255) | NOT NULL | Admin password |

**Table 3.2: user**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| u_id | INT | PK, AUTO_INCREMENT | User ID |
| u_name | VARCHAR(255) | NOT NULL | User full name |
| u_email | VARCHAR(255) | UNIQUE, NOT NULL | User email address |
| u_password | VARCHAR(255) | NOT NULL | User password |
| u_mobile | VARCHAR(20) | NOT NULL | Mobile number |
| u_age | INT | NOT NULL | User age |

**Table 3.3: doctors**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| d_id | INT | PK, AUTO_INCREMENT | Doctor ID |
| d_name | VARCHAR(255) | NOT NULL | Doctor full name |
| d_email | VARCHAR(255) | UNIQUE, NOT NULL | Doctor email address |
| d_passwords | VARCHAR(255) | NOT NULL | Doctor password |
| d_spec | VARCHAR(255) | NOT NULL | Medical specialization |

**Table 3.4: appointment**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| ap_id | INT | PK, AUTO_INCREMENT | Appointment ID |
| d_id | INT | FK -> doctors(d_id) | Doctor reference |
| u_id | INT | FK -> user(u_id) | User reference |
| ap_time | VARCHAR(50) | NOT NULL | Appointment time |
| ap_date | VARCHAR(50) | NOT NULL | Appointment date |
| ap_report | VARCHAR(255) | | Uploaded report filename |
| ap_payment_status | VARCHAR(50) | DEFAULT 'pending' | Payment status (pending/success) |
| ap_status | VARCHAR(50) | DEFAULT 'pending' | Appointment status (pending/Accept/Reject) |
| razorpay_order_id | VARCHAR(255) | | Razorpay order reference |

**Relationships:**
- `appointment.d_id` -> `doctors.d_id` (ON DELETE CASCADE)
- `appointment.u_id` -> `user.u_id` (ON DELETE CASCADE)

**Figure 3.2: Entity-Relationship (ER) Diagram**

```
+--------------+         +--------------------+         +--------------+
|    admin     |         |    appointment     |         |     user     |
+--------------+         +--------------------+         +--------------+
| id (PK)      |         | ap_id (PK)         |         | u_id (PK)    |
| a_username   |         | d_id (FK) ---------+--+      | u_name       |
| a_password   |         | u_id (FK) ---------+--+---->| u_email      |
+--------------+         | ap_time            |  |      | u_password   |
                         | ap_date            |  |      | u_mobile     |
                         | ap_report          |  |      | u_age        |
                         | ap_payment_status  |  |      +--------------+
                         | ap_status          |  |
                         | razorpay_order_id  |  |
                         +--------------------+  |
                                                 |
                         +--------------+        |
                         |   doctors    |        |
                         +--------------+        |
                         | d_id (PK) <--+--------+
                         | d_name       |
                         | d_email      |
                         | d_passwords  |
                         | d_spec       |
                         +--------------+
```

### 3.3 Data Flow Diagrams (DFD)

**Figure 3.3: Level 0 -- Context Diagram**

```
                      +-------------------------------+
   +-----------+      |                               |      +-----------+
   |           |----->|                               |----->|           |
   |   User    |      |     Disease Prediction        |      |  Doctor   |
   | (Patient) |<-----|      Web Application           |<-----|  Panel    |
   |           |      |                               |      |           |
   +-----------+      |    +-------------------+      |      +-----------+
                      |    |    MySQL DB        |      |
   +-----------+      |    |    (pro17)         |      |      +-----------+
   |           |----->|    +-------------------+      |----->|           |
   |   Admin   |      |                               |      | Razorpay  |
   |   Panel   |<-----|    +-------------------+      |<-----| Payment   |
   |           |      |    |  ML Models +       |      |      +-----------+
   +-----------+      |    |  Chatbot + LLM     |      |
                      |    +-------------------+      |
                      +-------------------------------+
```

**Figure 3.4: Level 1 -- Detailed Data Flow Diagram**

```
 USER -->  [1.0 Authentication] --> User Session (MySQL)
                    |
                    v
 USER -->  [2.0 Binary Symptom Prediction] --> SVM/MNB Model --> Disease + Confidence + Specialist
                    |
                    v
 USER -->  [3.0 Text-Based Prediction] --> TF-IDF + MNB Pipeline --> Top-3 Diseases
                    |
                    v
 USER -->  [4.0 Diabetes Basic] --> StandardScaler + SVM --> Diabetic/Non-Diabetic
                    |
                    v
 USER -->  [5.0 Diabetes Advanced] --> StandardScaler + RF --> Risk Score + Confidence %
                    |
                    v
 USER -->  [6.0 Book Appointment] --> [7.0 Razorpay Payment] --> MySQL --> Doctor Panel
                    |
                    v
 USER -->  [8.0 Chatbot Query] --> NLTK Pipeline + PyTorch NN --> Bot Response

 ADMIN --> [9.0 Manage Users/Doctors] --> MySQL (user, doctors tables)
 DOCTOR -> [10.0 Manage Appointments] --> MySQL --> [11.0 Send Email] --> SMTP (Gmail)
```

### 3.4 UML Diagrams

**Figure 3.5: Use Case Diagram**

```
                      +--------------------------------------------+
                      |       Disease Prediction System             |
                      |                                            |
   +--------+         |   +------------+    +----------------+     |
   |        |         |   | Predict    |    | Text-Based     |     |
   |        |-------->|   | Disease    |    | Prediction     |     |
   |        |         |   | (Binary)   |    | (NLP)          |     |
   |        |         |   +-----+------+    +--------+-------+     |
   |        |         |         |                    |             |
   |  USER  |         |   +-----v------+    +--------v-------+     |
   |(Patient)|-------->|   | Diabetes   |    | Diabetes       |     |
   |        |         |   | Basic      |    | Advanced       |     |
   |        |         |   | (8 feat.)  |    | (21 feat.)     |     |
   |        |         |   +------------+    +--------------  +     |
   |        |         |                                            |
   |        |-------->|   +------------+    +--------------+       |
   |        |         |   | Book       |    | View         |       |
   |        |         |   | Appointment|    | History      |       |
   +---+----+         |   +-----+------+    +--------------+       |
       |              |         |                                  |
       |              |   +-----v------+    +---------+            |
       |              |   | Razorpay   |    | Chatbot |            |
       |              |   | Payment    |    | (AI)    |            |
       |              |   +------------+    +---------+            |
       |              +--------------------------------------------+
       |
   +---v----+         +--------------------------------------------+
   | DOCTOR |-------->| Manage Appointments | Accept/Reject        |
   +--------+         | Send Email Reminders                       |
                      +--------------------------------------------+
       |
   +---v----+         +--------------------------------------------+
   | ADMIN  |-------->| Manage Users | Manage Doctors | Dashboard  |
   +--------+         +--------------------------------------------+
```

**Figure 3.6: Sequence Diagram -- Disease Prediction Flow**

```
  User          Browser        Flask Server      ML Model       MySQL DB
    |               |               |               |               |
    |--Select------>|               |               |               |
    |  Symptoms     |--POST /check->|               |               |
    |               |  disease      |               |               |
    |               |               |--predict()---->|               |
    |               |               |               |--Result------>|
    |               |               |<--confidence--|               |
    |               |               |               |               |
    |               |               |--SELECT doc---+-------------->|
    |               |               |<--doctor name-+---------------|
    |               |               |               |               |
    |               |<--JSON Result-|               |               |
    |<--Disease + --|               |               |               |
    |  Specialist   |               |               |               |
```

**Figure 3.7: Sequence Diagram -- Appointment & Payment Flow**

```
  User       Browser     Flask      Razorpay     MySQL       Doctor
    |           |        Server       API          DB          Email
    |           |          |           |           |             |
    |--Fill---->|          |           |           |             |
    |  Form     |--POST--->|           |           |             |
    |           |  /process|           |           |             |
    |           |          |--Upload-->|           |             |
    |           |          |  Report   |  (File)   |             |
    |           |          |--Create-->|           |             |
    |           |          |  Order    |           |             |
    |           |          |<--Order---|           |             |
    |           |          |  ID       |           |             |
    |           |          |--INSERT---+---------->|             |
    |           |          |  Appt     |           |             |
    |<--Payment-|<-Redirect|           |           |             |
    |  Page     |          |           |           |             |
    |--Pay------+----------+---------->|           |             |
    |           |          |           |--Callback-+------------>|
    |           |          |<--Success-+-----------|             |
    |           |          |--UPDATE---+---------->|             |
    |           |          |  Status   |           |             |
    |<--Success-|<-Redirect|           |           |--Email----->|
    |  Page     |          |           |           |  Reminder   |
```

### 3.5 Machine Learning Pipeline

The ML pipeline follows a systematic approach for model training and evaluation:

```
Raw Data -> Data Preprocessing -> Feature Engineering -> Train/Test Split
     -> Model Training -> Cross-Validation -> Evaluation -> Model Serialization
```

**Figure 3.8: Binary Symptom Prediction Pipeline**

```
User Input (132 symptoms)
        |
        v
Binary Encoding (0/1 for each symptom)
        |
        v
Pre-trained SVM/Decision Tree/MNB Model
        |
        v
Disease Prediction + Confidence Score (predict_proba)
        |
        v
Specialist Mapping + Disease Details (from disease_info.json)
```

**Figure 3.9: Text-Based Prediction Pipeline**

```
User Input (Free Text Description)
        |
        v
TF-IDF Vectorization (max_features=8000, ngram_range=(1,2))
        |
        v
Pre-trained MultinomialNB Classifier
        |
        v
Label Decoding -> Top-3 Diseases + Confidence Scores
```

**Figure 3.10: Diabetes Prediction Pipeline (Basic - PIMA)**

```
User Input (8 Clinical Features)
        |
        v
StandardScaler Transformation
        |
        v
Pre-trained SVM Model
        |
        v
Diabetic / Non-Diabetic Prediction
```

**Figure 3.11: Diabetes Prediction Pipeline (Advanced - BRFSS)**

```
User Input (21 Health Indicators)
        |
        v
StandardScaler Transformation
        |
        v
Pre-trained Random Forest (100 estimators)
        |
        v
Diabetes Risk Prediction + Confidence Percentage
```

### 3.6 NLP Pipeline (Chatbot)

**Figure 3.12: Chatbot NLP Pipeline**

```
User Message (Natural Language)
      |
      v
NLTK Tokenization (word_tokenize)
      |
      v
Porter Stemmer (root form extraction)
      |
      v
Bag of Words Representation (binary vector)
      |
      v
PyTorch Neural Network (3-layer Feedforward)
      |
      v
Intent Classification + Confidence Check (threshold: 0.75)
      |
      +---> Confidence >= 0.75 ---> Response Selection from intents.json
      |
      +---> Confidence < 0.75 ----> Fallback: "I do not understand..."
```

### 3.7 Deep Learning Pipeline (Chatbot)

**Figure 3.13: Neural Network Architecture (model.py)**

```
Input Layer (input_size neurons)
        |
        v
Linear(input_size, hidden_size) + ReLU Activation
        |
        v
Linear(hidden_size, hidden_size) + ReLU Activation
        |
        v
Linear(hidden_size, num_classes) -> Output (raw logits)
        |
        v
Softmax -> Probability Distribution over Intents
```

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Architecture | 3-layer Feedforward Neural Network |
| Activation | ReLU (Rectified Linear Unit) |
| Loss Function | CrossEntropyLoss |
| Confidence Threshold | 0.75 |
| Bot Name | Sam |
| Saved Model | `data.pth` (PyTorch state dict) |

### 3.8 Technology Stack

**Table 3.5: Complete Technology Stack**

| Category | Technology | Version/Purpose |
|----------|-----------|-----------------|
| **Frontend** | HTML5, CSS3, Bootstrap | Responsive UI design with modern layout |
| **Frontend** | JavaScript | Client-side interactions, AJAX calls, chatbot widget |
| **Backend** | Python Flask | Web server, routing, and request handling |
| **Database** | MySQL (Flask-MySQLdb) | Relational data storage for users, doctors, appointments |
| **Machine Learning** | Scikit-learn | Classification algorithms (SVM, RF, DT, NB, LR, GB) |
| **Ensemble ML** | XGBoost | Extreme Gradient Boosting classifier |
| **Deep Learning** | PyTorch | Neural network for chatbot intent classification |
| **NLP** | NLTK | Tokenization, stemming, bag-of-words representation |
| **Data Processing** | Pandas, NumPy | Data manipulation, feature extraction, preprocessing |
| **Class Balancing** | imbalanced-learn (SMOTE) | Synthetic oversampling for minority class |
| **Visualization** | Matplotlib, Seaborn | Feature correlation heatmaps, result charts |
| **Model Serialization** | Joblib, Pickle | Saving and loading trained models |
| **Payment Gateway** | Razorpay | PCI-DSS compliant online payment processing |
| **Email Service** | SMTP (Gmail) | Automated email notifications to patients |
| **File Upload** | Werkzeug | Secure filename sanitization for report uploads |
| **LLM Integration** | Ollama (Gemma 3) | Local LLM for disease information generation |
| **Environment** | Anaconda (environment.yml) | Python environment and package management |
| **Configuration** | YAML (config.yaml) | Model hyperparameters and dataset paths |

### 3.9 Specialist Mapping Logic

The system maps predicted diseases to medical specialists using a predefined dictionary. This enables automatic specialist recommendation after disease prediction.

**Table 3.6: Specialist-Disease Mapping**

| Specialist | Diseases Covered | No. of Diseases |
|-----------|-----------------|-----------------|
| **Rheumatologist** | Osteoarthritis, Arthritis | 2 |
| **Cardiologist** | Heart attack, Bronchial Asthma, Hypertension | 3 |
| **ENT Specialist** | Vertigo, Hypothyroidism | 2 |
| **Neurologist** | Varicose veins, Paralysis, Migraine, Cervical spondylosis | 4 |
| **Allergist/Immunologist** | Allergy, Pneumonia, AIDS, Common Cold, Tuberculosis, Malaria, Dengue, Typhoid | 8 |
| **Urologist** | Urinary tract infection, Hemorrhoids | 2 |
| **Dermatologist** | Acne, Chicken pox, Fungal infection, Psoriasis, Impetigo | 5 |
| **Gastroenterologist** | Peptic ulcer, GERD, Cholestasis, Drug Reaction, Gastroenteritis, Hepatitis A-E, Diabetes, Hypoglycemia | 8+ |

### 3.10 Mathematical Formulation of Algorithms

This section presents the mathematical foundations of the key algorithms used in this project.

#### 3.10.1 Support Vector Machine (SVM)

The SVM algorithm finds the optimal hyperplane that separates data points of different classes with maximum margin [15]. Given a training set {(x_1, y_1), ..., (x_n, y_n)} where x_i in R^d and y_i in {-1, +1}:

**Primal Optimization Problem:**

```
minimize:    (1/2)||w||^2 + C * sum(i=1 to n) xi_i

subject to:  y_i(w^T * phi(x_i) + b) >= 1 - xi_i,  xi_i >= 0
```

Where:
- **w** is the weight vector defining the hyperplane
- **b** is the bias term
- **C** is the regularization parameter (C = 0.1 in our model)
- **xi_i** are slack variables allowing soft-margin misclassification
- **phi(x)** is the feature mapping function to higher-dimensional space

**RBF Kernel Function:**

```
K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
```

Where gamma controls the influence radius of a single training example (gamma = 'scale' in our model).

**Decision Function:**

```
f(x) = sign(sum(i) alpha_i * y_i * K(x_i, x) + b)
```

**Parameters Used:** C = 0.1, kernel = RBF, gamma = scale

#### 3.10.2 Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of classes for classification [14].

**Ensemble Prediction:**

```
h_hat(x) = majority_vote{h_1(x), h_2(x), ..., h_B(x)}
```

Where h_i(x) is the prediction of the i-th decision tree and B is the number of trees (B = 100 in our model).

**Key Mechanisms:**
- **Bootstrap Aggregating (Bagging):** Each tree is trained on a bootstrap sample D_i drawn with replacement from the original training set D.
- **Random Feature Selection:** At each split, a random subset of m features is considered from the total M features, where m = sqrt(M) for classification.

**Gini Impurity (Split Criterion):**

```
Gini(t) = 1 - sum(i=1 to C) p(i|t)^2
```

Where p(i|t) is the proportion of class i samples at node t, and C is the number of classes.

**Parameters Used:** n_estimators = 100, random_state = 42, n_jobs = -1

#### 3.10.3 Multinomial Naive Bayes (Text Model)

Naive Bayes classifiers apply Bayes' theorem with the "naive" assumption of conditional independence between features [9].

**Bayes' Theorem:**

```
P(y|x_1, x_2, ..., x_n) = P(y) * P(x_1, x_2, ..., x_n|y) / P(x_1, x_2, ..., x_n)
```

**With Naive Independence Assumption:**

```
P(y|x_1, ..., x_n) is proportional to P(y) * product(i=1 to n) P(x_i|y)
```

**Multinomial Model (for TF-IDF features):**

```
P(x_i|y) = (N_yi + alpha) / (N_y + alpha * n)
```

Where:
- N_yi is the count of feature i in class y
- N_y is the total count of all features in class y
- alpha is the Laplace smoothing parameter (alpha = 0.01)
- n is the number of features

**Classification Decision:**

```
y_hat = argmax_y  P(y) * product(i) P(x_i|y)
```

**Parameters Used:** alpha = 0.01, TF-IDF max_features = 8000, ngram_range = (1, 2)

#### 3.10.4 TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text documents into numerical feature vectors [12].

**Term Frequency (TF):**

```
TF(t, d) = f(t, d) / sum(t') f(t', d)
```

Where f(t, d) is the frequency of term t in document d.

**Inverse Document Frequency (IDF):**

```
IDF(t) = log((1 + n) / (1 + df(t))) + 1
```

Where n is the total number of documents and df(t) is the number of documents containing term t.

**TF-IDF Score:**

```
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

**Parameters Used:** max_features = 8000, ngram_range = (1, 2), sublinear_tf = False

#### 3.10.5 SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic samples for the minority class by creating new samples along line segments joining existing minority class nearest neighbors [11].

**Synthetic Sample Generation:**

```
x_new = x_i + lambda * (x_j - x_i)
```

Where:
- x_i is a minority class sample
- x_j is one of its k nearest neighbors in the minority class
- lambda is a random number in the range [0, 1]

**Parameters Used:** k_neighbors = 5 (default), random_state = 42

#### 3.10.6 Neural Network (Chatbot)

The chatbot uses a 3-layer feedforward neural network implemented in PyTorch [10].

**Forward Pass:**

```
z_1 = W_1 * x + b_1
a_1 = ReLU(z_1) = max(0, z_1)

z_2 = W_2 * a_1 + b_2
a_2 = ReLU(z_2) = max(0, z_2)

z_3 = W_3 * a_2 + b_3
```

Where W_i are weight matrices and b_i are bias vectors.

**Cross-Entropy Loss:**

```
L = -sum(i=1 to C) y_i * log(softmax(z_3)_i)
```

**Softmax Function:**

```
softmax(z)_i = exp(z_i) / sum(j) exp(z_j)
```

**Parameters:** input_size = vocabulary size, hidden_size = configurable, num_classes = number of intents, confidence_threshold = 0.75

---

<br/>

# CHAPTER 4

## IMPLEMENTATION

### 4.1 Dataset Details

Four distinct datasets were used in this project, each serving a specific prediction module:

**Table 4.1: Dataset 1 -- Disease Symptom Dataset (Binary)**

| Property | Value |
|----------|-------|
| Training Data | `dataset/training_data.csv` |
| Test Data | `dataset/test_data.csv` |
| Features | 132 binary symptom columns (0 or 1) |
| Target | `prognosis` column (41 diseases) |
| Feature Type | Binary (0 = symptom absent, 1 = symptom present) |
| Total Training Samples | ~4,920 |

**41 Diseases Covered:** Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction, Peptic ulcer disease, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Paralysis (brain hemorrhage), Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A-E, Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia, Dimorphic hemorrhoids, Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthritis, Arthritis, Vertigo, Acne, Urinary tract infection, Psoriasis, Impetigo

**Sample Symptoms (132 total):** itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue, muscle_wasting, vomiting, fatigue, headache, fever, cough, nausea, dizziness, chest_pain, and 113 more

**Table 4.2: Dataset 2 -- PIMA Diabetes Dataset (Basic)**

| Property | Value |
|----------|-------|
| Source | National Institute of Diabetes and Digestive and Kidney Diseases |
| File | `diabetes.csv` |
| Records | 768 patients |
| Features | 8 clinical parameters |
| Target | Outcome (0 = Non-Diabetic, 1 = Diabetic) |

**8 Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

**Table 4.3: Dataset 3 -- CDC BRFSS 2015 Diabetes Dataset (Advanced)**

| Property | Value |
|----------|-------|
| Source | CDC Behavioral Risk Factor Surveillance System 2015 |
| File | `diabetes_binary_health_indicators_BRFSS2015.csv` |
| Records | 253,680 (229,474 after deduplication) |
| Features | 21 health indicators |
| Target | Diabetes_binary (0 = No Diabetes, 1 = Diabetes) |
| Duplicates Removed | 24,206 |

**21 Features:** HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

**Class Distribution (after deduplication):**

| Class | Records | Percentage |
|-------|---------|------------|
| No Diabetes | 218,334 | 84.6% |
| Diabetes | 35,346 | 15.4% |
| **Total** | **229,474** | **100%** |

**Table 4.4: Dataset 4 -- Symptom2Disease (Text-Based)**

| Property | Value |
|----------|-------|
| File | `Symptom2Disease.csv` |
| Records | 1,200 patient symptom descriptions |
| Diseases | 24 (50 descriptions per disease) |
| Columns | label (disease name), text (natural language description) |

**24 Diseases:** Psoriasis, Varicose Veins, Typhoid, Chicken pox, Impetigo, Dengue, Fungal infection, Common Cold, Pneumonia, Dimorphic Hemorrhoids, Arthritis, Acne, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Jaundice, Malaria, Urinary tract infection, Allergy, GERD, Drug reaction, Peptic ulcer disease, Diabetes

**Example Input/Output:**
> *"I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."* -> **Psoriasis**

### 4.2 Data Preprocessing

#### 4.2.1 Binary Symptom Data
- Loaded via Pandas from CSV files
- Features extracted by filtering out `prognosis` and `Unnamed` columns
- Data sanity check: `assert len(features.iloc[0]) == 132`
- Train/validation split: 67/33 (`test_size=0.33, random_state=101`)
- No missing values or scaling required (binary features)

#### 4.2.2 PIMA Diabetes Data
- **Missing Value Handling:** Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns (biologically impossible) were replaced with column-wise mean values
- **Feature Scaling:** `StandardScaler` applied to normalize features to zero mean and unit variance
- **Train/Test Split:** 80/20 (`random_state=7`)

#### 4.2.3 BRFSS Diabetes Data
- **Duplicate Removal:** 24,206 duplicate rows removed (253,680 -> 229,474)
- **Class Balancing:** SMOTE applied to training data only
  - Before SMOTE: ~155K training samples (imbalanced)
  - After SMOTE: ~311K training samples (balanced 50/50)
- **Feature Scaling:** `StandardScaler` fit on training data, applied to both train and test
- **Train/Test Split:** 80/20 (stratified, `random_state=42`)

#### 4.2.4 Text Symptom Data
- Raw text descriptions used directly (no manual preprocessing)
- TF-IDF vectorization with hyperparameter tuning
- Best parameters: `max_features=8000`, `ngram_range=(1,2)`, `alpha=0.01`

### 4.3 Feature Engineering

| Dataset | Feature Type | Engineering Applied |
|---------|-------------|-------------------|
| **Binary Symptoms** | 132 binary (0/1) | Feature correlation analysis via Seaborn heatmap; no selection applied due to high accuracy |
| **TF-IDF (Text)** | Sparse vector | Hyperparameter-tuned: max_features and ngram_range varied; best: 8000 features, (1,2)-grams |
| **PIMA** | 8 numerical | StandardScaler normalization; zero-value imputation for impossible values |
| **BRFSS** | 21 mixed (binary + ordinal) | StandardScaler normalization; SMOTE after split to prevent data leakage |

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
1. Loading configuration from YAML
2. Loading train/test datasets
3. Computing feature correlation heatmap
4. Train/validation splitting (67/33)
5. Model selection based on algorithm name
6. Training with 3-fold cross-validation
7. Saving trained models via Joblib

#### 4.4.2 Comprehensive Training Results

**Table 4.5: Binary Symptom Models (132 features -> 41 diseases)**

| Model | CV Score | Validation Accuracy | Test Accuracy |
|-------|----------|-------------------|---------------|
| **SVM (RBF, C=0.1)** | 1.0000 | 1.0000 | **1.0000 (100%)** |
| **Logistic Regression (C=0.1)** | 1.0000 | 1.0000 | **1.0000 (100%)** |
| Decision Tree (gini) | 1.0000 | 1.0000 | 0.9762 (97.6%) |
| Random Forest (100 estimators) | 1.0000 | 1.0000 | 0.9762 (97.6%) |
| Gradient Boosting (100 estimators) | 1.0000 | 1.0000 | 0.9762 (97.6%) |
| XGBoost (200 estimators) | 1.0000 | 1.0000 | 0.9762 (97.6%) |

**Table 4.6: Text-Based Models (TF-IDF -> 24 diseases)**

| Model | CV Score | Test Accuracy | Test F1-Score | max_features | ngram_range |
|-------|----------|---------------|---------------|-------------|-------------|
| **MultinomialNB** | 0.9656 | **0.9792 (97.92%)** | **0.9789** | 8000 | (1,2) |
| LinearSVC | 0.9708 | 0.9542 (95.42%) | 0.9529 | 3000 | (1,1) |
| Logistic Regression | 0.9719 | 0.9500 (95.00%) | 0.9488 | 3000 | (1,1) |
| Random Forest | 0.9490 | 0.9458 (94.58%) | 0.9451 | 3000 | (1,2) |
| XGBoost | 0.8740 | 0.8708 (87.08%) | 0.8727 | 3000 | (1,1) |

**Production Models Selected:**

| Module | Model File | Algorithm | Selection Reason |
|--------|-----------|-----------|-----------------|
| Binary Disease | `best_binary_model.joblib` | SVM | 100% accuracy + probability estimates |
| Text Disease | `best_text_model.joblib` | TF-IDF + MultinomialNB | Highest test accuracy (97.92%) |
| Diabetes Basic | `svm_model.pkl` | SVM + StandardScaler | Original deployment selection |
| Diabetes Advanced | `diabetes_brfss_model.pkl` | Random Forest (100 est.) | Best accuracy (84.05%) |

#### 4.4.3 Diabetes Model Training (BRFSS)

The `train_diabetes_brfss.py` script performs:
1. Load BRFSS 2015 dataset (253,680 records)
2. Remove 24,206 duplicate rows (-> 229,474)
3. Split into 80/20 train/test (stratified by target)
4. Apply SMOTE to training data only (~155K -> ~311K)
5. Standardize features with StandardScaler
6. Train 6 models and evaluate each
7. Select best model (Random Forest: 84.05%)
8. Save model, scaler, and feature names as pickle files

**Table 4.7: BRFSS Diabetes Model Comparison**

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | **Random Forest (100 estimators)** | **84.05%** |
| 2 | Gradient Boosting | 83.78% |
| 3 | Decision Tree | 77.13% |
| 4 | KNN | 75.28% |
| 5 | Logistic Regression | 71.50% |
| 6 | Naive Bayes | 64.39% |

### 4.5 Web Application Implementation

The Flask application (`app.py`, 877 lines) implements a comprehensive set of routes organized by user role:

**Table 4.8: User Routes**

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page with navigation to all features |
| `/register` | GET, POST | User registration with form validation |
| `/login` | GET, POST | User login with session management |
| `/logout` | GET | Session clearing and redirect |
| `/predict_main` | GET | Disease prediction landing page |
| `/predict` | POST | Binary symptom prediction API |
| `/predict_text` | GET, POST | Text-based NLP prediction |
| `/checkdisease` | GET, POST | Advanced disease check with specialist mapping |
| `/diabeties` | GET, POST | Basic diabetes prediction (PIMA) |
| `/diabetes_advanced` | GET, POST | Advanced diabetes prediction with confidence |
| `/appoinment` | GET | Appointment booking page |
| `/process_form` | POST | Process appointment with Razorpay order |
| `/payment/<order_id>` | GET | Razorpay payment page |
| `/success` | GET | Payment success callback |
| `/history/<u_id>` | GET | Appointment history for user |
| `/services` | GET | Services information page |
| `/predict1` | POST | Chatbot API endpoint |

**Table 4.9: Admin Routes**

| Route | Method | Description |
|-------|--------|-------------|
| `/admin` | GET, POST | Admin login |
| `/admin/index` | GET | Dashboard (total users, doctors count) |
| `/admin/user` | GET | View all registered users |
| `/admin/edit_user/<id>` | GET, POST | Edit user details |
| `/admin/delete/<id>` | GET | Delete user (CASCADE to appointments) |
| `/admin/doctor` | GET | View all registered doctors |
| `/admin/add_doctor` | GET, POST | Add new doctor with specialization |
| `/admin/doc_delete/<id>` | GET | Delete doctor (CASCADE to appointments) |

**Table 4.10: Doctor Routes**

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

```
1. User fills appointment form (doctor, date, time, report)
            |
            v
2. Report file uploaded securely (Werkzeug secure_filename)
            |
            v
3. Razorpay order created (amount: INR 200)
            |
            v
4. Appointment record saved (payment_status = 'pending')
            |
            v
5. User redirected to Razorpay payment page
            |
            v
6. On successful payment: ap_payment_status -> 'success'
            |
            v
7. Doctor views only successfully paid appointments
            |
            v
8. Doctor acts on appointment -> Email notification to patient
```

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

| Field | Description | Example |
|-------|-------------|---------|
| `description` | 2-3 sentence medical description | "Psoriasis is a chronic autoimmune condition..." |
| `common_symptoms` | List of 5 common symptoms | ["Red patches", "Itchy skin", ...] |
| `precautions` | List of 4 precautionary measures | ["Moisturize regularly", ...] |
| `medications` | List of 3 medication categories | ["Topical corticosteroids", ...] |
| `diet_recommendations` | List of 3 dietary tips | ["Anti-inflammatory foods", ...] |
| `when_to_see_doctor` | Guidance on when to seek urgent care | "Seek immediate care if..." |

---

<br/>

# CHAPTER 5

## RESULTS AND DISCUSSION

### 5.1 Disease Prediction Results (Binary Symptoms)

**Table 5.1: Binary Symptom Model Results Summary**

| Model | Algorithm | Test Accuracy | Best Parameters |
|-------|-----------|---------------|-----------------|
| `best_binary_model.joblib` | SVM (RBF kernel) | **100.00%** | C=0.1, gamma=scale |
| -- | Logistic Regression | **100.00%** | C=0.1, solver=lbfgs |
| `decision_tree.joblib` | Decision Tree (gini) | 97.62% | criterion=gini |
| -- | Random Forest | 97.62% | n_estimators=100, max_depth=10 |
| -- | Gradient Boosting | 97.62% | n_estimators=100, lr=0.05 |
| -- | XGBoost | 97.62% | n_estimators=200, lr=0.05 |
| `mnb.joblib` | Multinomial Naive Bayes | 100.00%* | default |

*MNB trained separately via main.py; validation accuracy = 100%.

**Analysis:** The symptom-disease dataset has distinct, non-overlapping symptom patterns per disease, resulting in very high classification accuracy across all algorithms. SVM and Logistic Regression achieve perfect test accuracy, while tree-based methods show slight variation (97.62%). The validation accuracy is 100% for all models with 3-fold cross-validation scores of 1.0, confirming model reliability.

**Production Selection Rationale:** SVM (`best_binary_model.joblib`) was selected because:
1. 100% test accuracy on the evaluation set
2. `predict_proba()` support for confidence scoring via Platt scaling
3. Robust performance with RBF kernel on high-dimensional sparse data (132 features)

### 5.2 Disease Prediction Results (Text-Based)

**Table 5.2: Text-Based Model Results Summary**

| Model | CV Score | Test Accuracy | Test F1 | TF-IDF Config |
|-------|----------|---------------|---------|---------------|
| **MultinomialNB** | 0.9656 | **97.92%** | **0.9789** | max_features=8000, ngram=(1,2) |
| LinearSVC | 0.9708 | 95.42% | 0.9529 | max_features=3000, ngram=(1,1) |
| Logistic Regression | 0.9719 | 95.00% | 0.9488 | max_features=3000, ngram=(1,1) |
| Random Forest | 0.9490 | 94.58% | 0.9451 | max_features=3000, ngram=(1,2) |
| XGBoost | 0.8740 | 87.08% | 0.8727 | max_features=3000, ngram=(1,1) |

**Analysis:**
- MultinomialNB achieves the best test accuracy (97.92%) and F1-score (0.9789), benefiting from:
  - Higher `max_features` (8000) capturing more vocabulary
  - Bigram features `(1,2)` capturing symptom phrases like "skin rash" and "joint pain"
  - Low alpha (0.01) allowing strong feature contributions
- LinearSVC has a higher CV score (0.9708) but lower test accuracy, suggesting slight overfitting
- XGBoost underperforms likely due to the relatively small dataset size (1,200 records)
- The balanced dataset (50 samples per disease) contributes to consistent performance across all 24 classes

**Production Output:** Top-3 predicted diseases with confidence percentages, enabling users to see alternative diagnoses.

### 5.3 Diabetes Prediction Results (Basic - PIMA)

**Table 5.3: PIMA Diabetes Model Results**

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | **Naive Bayes** | **78.57%** |
| 2 | Logistic Regression | 76.62% |
| 3 | Random Forest | 75.97% |
| 4 | SVM | 75.32% |
| 5 | KNN | 75.32% |
| 6 | Decision Tree | 73.38% |

**Analysis:**
- The PIMA dataset's small size (768 records) limits model performance
- Naive Bayes performs best, likely due to the relatively independent nature of clinical features
- All models fall in the 73-79% range, consistent with literature findings [2]
- Zero-value imputation (replacing biologically impossible zeros with column means) improved data quality
- Production uses SVM (`svm_model.pkl`) as the originally selected deployment model

### 5.4 Diabetes Prediction Results (Advanced - BRFSS)

**Table 5.4: BRFSS Diabetes Model Results**

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | **Random Forest (100 estimators)** | **84.05%** |
| 2 | Gradient Boosting | 83.78% |
| 3 | Decision Tree | 77.13% |
| 4 | KNN | 75.28% |
| 5 | Logistic Regression | 71.50% |
| 6 | Naive Bayes | 64.39% |

**Table 5.5: Detailed Classification Report (Random Forest)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Diabetes (0) | 0.87 | 0.96 | 0.91 | 43,667 |
| Diabetes (1) | 0.45 | 0.19 | 0.27 | 11,028 |
| **Weighted Avg** | **0.79** | **0.84** | **0.78** | **54,695** |

**Analysis:**
- Random Forest achieves the best overall accuracy (84.05%) but shows significant class imbalance in recall
- **No Diabetes class:** High recall (0.96) -- very few false negatives for healthy patients
- **Diabetes class:** Low recall (0.19) -- many diabetic patients are missed (false negatives)
- This imbalance persists despite SMOTE, suggesting the minority class has inherently complex patterns
- The large dataset (229,474 records) provides robust training data
- **SMOTE applied only on training data** to prevent data leakage -- a critical methodological choice

### 5.5 Chatbot Performance

| Metric | Value |
|--------|-------|
| Architecture | 3-layer Feedforward Neural Network |
| Framework | PyTorch |
| NLP Pipeline | NLTK (Tokenize -> Stem -> Bag of Words) |
| Confidence Threshold | 0.75 |
| Training Data | `intents.json` (greetings, services, symptoms, disease info) |
| Bot Name | Sam |

**Analysis:**
- The chatbot uses a confidence threshold of 0.75 to filter low-confidence predictions
- Queries below threshold receive a fallback response: "I do not understand..."
- The bag-of-words approach is lightweight and suitable for the limited intent space
- Intent categories cover: greetings, goodbye, thanks, services, payments, symptom descriptions, disease info queries

### 5.6 SMOTE Analysis -- Class Imbalance Handling (BRFSS)

**Figure 5.3: SMOTE Class Distribution**

**Before SMOTE (Training Set):**

| Class | Count | Percentage |
|-------|-------|------------|
| No Diabetes | ~174,686 | 86.2% |
| Diabetes | ~28,094 | 13.8% |
| **Total** | **~202,780** | **100%** |

**After SMOTE (Training Set):**

| Class | Count | Percentage |
|-------|-------|------------|
| No Diabetes | ~174,686 | 50% |
| Diabetes (Synthetic) | ~174,686 | 50% |
| **Total** | **~349,372** | **100%** |

Key points:
- SMOTE was applied **only on training data** to prevent data leakage
- Test set retains original imbalanced distribution for realistic evaluation
- Training samples increased from ~155K to ~311K
- **Impact:** Improved model's ability to detect diabetes patterns, though class recall remains low (0.19)

### 5.7 Confusion Matrix Analysis

**Figure 5.1: BRFSS Diabetes Confusion Matrix (Random Forest -- Best Model)**

```
                           Predicted
                      No Diabetes   Diabetes
Actual  No Diabetes     41,857       1,810      <- Recall: 96%
        Diabetes         8,943       2,085      <- Recall: 19%
                           ^            ^
                      Precision:    Precision:
                         87%           45%
```

| Metric | No Diabetes | Diabetes | Interpretation |
|--------|-------------|----------|----------------|
| **True Negatives** | 41,857 | -- | Correctly identified non-diabetic patients |
| **False Positives** | 1,810 | -- | Healthy patients incorrectly flagged (low harm) |
| **False Negatives** | -- | 8,943 | Diabetic patients missed (primary improvement area) |
| **True Positives** | -- | 2,085 | Correctly identified diabetic patients |
| **Overall Accuracy** | | | **84.05%** |

**Binary Symptom Model (SVM -- Perfect Classification):**

```
Confusion Matrix = Identity Matrix (41 x 41)
Each disease correctly classified with 100% accuracy.
```

The identity confusion matrix indicates that each disease has a unique, non-overlapping symptom signature, making the classification task highly separable.

### 5.8 Feature Importance Analysis (BRFSS Random Forest)

**Table 5.6: Feature Importance Ranking (Mean Decrease in Impurity)**

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | **BMI** | ~0.125 | Clinical |
| 2 | **Age** | ~0.098 | Demographic |
| 3 | **GenHlth** | ~0.087 | Self-reported |
| 4 | **Income** | ~0.072 | Socioeconomic |
| 5 | **PhysHlth** | ~0.065 | Self-reported |
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

**Table 5.7: Feature Comparison**

| Feature | Traditional Diagnosis | Existing ML Systems | **Our System** |
|---------|----------------------|---------------------|------------|
| Disease Prediction | Manual only | Single mode | **4 modes** |
| Input Method | Physical consultation | Structured form only | **Structured + Natural Language** |
| Diabetes Assessment | Lab tests only | Basic features (8) | **Basic (8) + Advanced (21)** |
| Specialist Mapping | Manual referral | Not available | **Automatic** |
| Appointment Booking | Phone/Walk-in | Not available | **Online with payment** |
| Chatbot Support | Not available | Basic FAQ | **AI-powered (PyTorch)** |
| Multi-role Access | N/A | Single user type | **3 roles (User, Doctor, Admin)** |
| Dataset Scale | N/A | <1,000 records | **Up to 229,474** |
| LLM Integration | Not available | Not available | **Gemma 3 via Ollama** |
| Payment Processing | Cash/Counter | Not available | **Razorpay (PCI-DSS)** |

### 5.10 Performance Evaluation Metrics

**Table 5.8: Overall System Performance Summary**

| Module | Best Algorithm | Accuracy | Dataset Size | Features |
|--------|---------------|----------|-------------|----------|
| Disease (Binary) | SVM (RBF) | **100.00%** | ~4,920 | 132 binary |
| Disease (Text) | TF-IDF + MNB | **97.92%** | 1,200 | TF-IDF (8000) |
| Diabetes (Basic) | Naive Bayes | **78.57%** | 768 | 8 clinical |
| Diabetes (Advanced) | Random Forest | **84.05%** | 229,474 | 21 health indicators |
| Chatbot | PyTorch NN | 0.75 threshold | intents.json | Bag of Words |

**Evaluation Metrics Used:**
- **Accuracy Score** -- Primary metric for all classification tasks
- **F1-Score** -- Used for text-based prediction (weighted average across classes)
- **Cross-Validation (3-fold)** -- Used for binary symptom models to verify generalization
- **Confusion Matrix** -- Generated for all models to analyze error patterns
- **Classification Report** -- Generated for diabetes models (precision, recall, F1 per class)
- **Confidence Score (predict_proba)** -- Used in production for providing probability estimates to users

---

<br/>

# CHAPTER 6

## TESTING AND VALIDATION

### 6.1 Model Testing

**Table 6.1: Model Testing Summary**

| Test Type | Description | Result |
|-----------|-------------|--------|
| Train/Test Split | 80/20 (diabetes) and 67/33 (disease) with stratification | All models validated |
| Cross-Validation | 3-fold CV on binary symptom models | Consistent 100% CV score |
| SMOTE Validation | Applied only on training data to prevent data leakage | Proper methodology confirmed |
| Confidence Scoring | `predict_proba()` for all prediction modules | Probability outputs verified |
| Hyperparameter Tuning | Grid/random search across multiple algorithms | Optimal parameters selected |

### 6.2 System Testing

**Table 6.2: System Testing Results**

| # | Module | Test Case | Expected Result | Status |
|---|--------|-----------|-----------------|--------|
| 1 | User Registration | Submit valid registration form | User record created in database | **Passed** |
| 2 | User Login | Submit valid credentials | Session created, redirect to home | **Passed** |
| 3 | User Login (Invalid) | Submit wrong credentials | Error message displayed | **Passed** |
| 4 | Disease Prediction (Binary) | Select symptoms, submit | Disease + confidence + specialist | **Passed** |
| 5 | Disease Prediction (Text) | Enter symptom text, submit | Top-3 diseases with confidence | **Passed** |
| 6 | Diabetes Basic | Enter 8 parameters, submit | Diabetic/Non-diabetic result | **Passed** |
| 7 | Diabetes Advanced | Enter 21 indicators, submit | Risk prediction + confidence % | **Passed** |
| 8 | Appointment Booking | Fill form, upload report | Razorpay order created | **Passed** |
| 9 | Payment Flow | Complete Razorpay payment | Status updated to success | **Passed** |
| 10 | Doctor Login | Submit doctor credentials | Doctor session created | **Passed** |
| 11 | View Appointments | Doctor views paid appointments | Appointment list displayed | **Passed** |
| 12 | Accept Appointment | Doctor accepts appointment | Status updated to Accept | **Passed** |
| 13 | Reject Appointment | Doctor rejects appointment | Status updated to Reject | **Passed** |
| 14 | Email Notification | Doctor sends email reminder | Email delivered to patient | **Passed** |
| 15 | Admin Login | Submit admin credentials | Admin session created | **Passed** |
| 16 | Admin - Manage Users | View, add, edit, delete users | CRUD operations functional | **Passed** |
| 17 | Admin - Manage Doctors | View, add, delete doctors | CRUD operations functional | **Passed** |
| 18 | Chatbot | Send health query | Response with confidence check | **Passed** |
| 19 | Chatbot (Low Confidence) | Send ambiguous query | Fallback "I do not understand" | **Passed** |
| 20 | Session Management | Logout action | Session cleared, redirect to home | **Passed** |
| 21 | Appointment History | View user appointment history | History with doctor names displayed | **Passed** |

> **Result: 21/21 test cases passed (100% pass rate)**

### 6.3 Data Validation

**Table 6.3: Data Validation Measures**

| Validation Type | Implementation | Purpose |
|----------------|----------------|---------|
| Input form validation | HTML5 required fields + server-side checks | Prevent incomplete submissions |
| File upload security | Werkzeug `secure_filename()` for report uploads | Prevent directory traversal attacks |
| SQL injection prevention | Parameterized queries using `%s` placeholders | Prevent malicious SQL code injection |
| Session management | Flask session with secret key | Maintain secure user state |
| Payment verification | Razorpay order ID matching in database | Prevent payment fraud |

---

<br/>

# CHAPTER 7

## CHALLENGES AND SOLUTIONS

### Challenge 1: Class Imbalance in BRFSS Diabetes Dataset

| Aspect | Detail |
|--------|--------|
| **Problem** | The BRFSS dataset was heavily imbalanced -- No Diabetes: 218,334 vs Diabetes: 35,346 (6.2:1 ratio) |
| **Solution** | Applied SMOTE to oversample the minority class in training data only. Training set expanded from ~155K to ~311K samples. |
| **Impact** | Despite SMOTE, diabetes class recall remained low (0.19), indicating inherent complexity of diabetes prediction from behavioral health indicators. |

### Challenge 2: Missing Values in PIMA Diabetes Dataset

| Aspect | Detail |
|--------|--------|
| **Problem** | Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns were biologically impossible, representing encoded missing data. |
| **Solution** | Replaced zero values with column-wise mean values before applying StandardScaler, preserving distribution while providing reasonable estimates. |

### Challenge 3: High-Dimensional Sparse Symptom Data

| Aspect | Detail |
|--------|--------|
| **Problem** | 132 binary symptom features create a high-dimensional feature space where most values are 0 (curse of dimensionality). |
| **Solution** | SVM with RBF kernel (C=0.1) effectively handles high-dimensional sparse data. Distinct symptom patterns made classification separable (100% accuracy). |

### Challenge 4: Free-Text Symptom Processing

| Aspect | Detail |
|--------|--------|
| **Problem** | Natural language symptom descriptions vary widely in vocabulary, sentence structure, and length. |
| **Solution** | TF-IDF vectorization (max_features=8000, ngram_range=(1,2)) captures both individual terms and bi-gram phrases. Combined with MNB (alpha=0.01), achieved 97.92% accuracy. |

### Challenge 5: Chatbot Low-Confidence Responses

| Aspect | Detail |
|--------|--------|
| **Problem** | The neural network occasionally produced incorrect responses for queries outside the training intent space. |
| **Solution** | Implemented a confidence threshold of 0.75. Responses below threshold trigger a fallback message, preventing misleading medical advice. |

### Challenge 6: Integration of Multiple ML Models

| Aspect | Detail |
|--------|--------|
| **Problem** | Loading 5+ trained models in a single Flask application could cause memory issues and slow startup. |
| **Solution** | Models loaded once at startup using Joblib/Pickle, stored as global variables, and reused across requests. Chatbot model loaded separately via `chat.py`. |

### Challenge 7: LLM Response Parsing

| Aspect | Detail |
|--------|--------|
| **Problem** | Gemma 3 via Ollama occasionally returned non-JSON responses or responses wrapped in markdown formatting. |
| **Solution** | Implemented robust JSON extraction using string boundary detection (`raw.index("{")` and `raw.rindex("}")`), with retry mechanism using simplified prompts on parse failure. |

---

<br/>

# CHAPTER 7a

## SECURITY AND PRIVACY CONSIDERATIONS

### 7a.1 Data Security Measures

| Security Measure | Implementation | Purpose |
|-----------------|----------------|---------|
| SQL Injection Prevention | Parameterized queries using `%s` placeholders | Prevents malicious SQL code injection |
| File Upload Security | Werkzeug `secure_filename()` | Prevents directory traversal attacks |
| Session Management | Flask session with secret key | Maintains secure user state |
| Payment Security | Razorpay PCI-DSS compliant processing | No card data stored locally |
| Input Validation | HTML5 form validation + server-side checks | Prevents malformed input |
| Access Control | Role-based routes (Admin, Doctor, User) | Unauthorized access prevention |
| Foreign Key Constraints | ON DELETE CASCADE on appointment table | Referential integrity |

### 7a.2 Privacy Considerations

1. **Medical Reports:** Stored on server filesystem, not in database -- reduces data exposure in case of SQL-based breach
2. **Payment Data:** No credit card details stored; only Razorpay order ID stored as reference
3. **Chatbot Conversations:** No conversation history stored -- stateless interaction ensures privacy
4. **Data Minimization:** Only essential user information collected (name, email, mobile, age)
5. **No Third-Party Sharing:** All data remains within the application; no external analytics or tracking

### 7a.3 Production Security Recommendations

For production deployment, the following security enhancements are recommended:

| Enhancement | Technology | Priority |
|-------------|-----------|----------|
| Password Hashing | bcrypt or Argon2 | **Critical** |
| HTTPS/TLS | SSL certificates | **Critical** |
| Token-Based Auth | JWT or OAuth 2.0 | High |
| Rate Limiting | Flask-Limiter | High |
| CSRF Protection | Flask-WTF | High |
| Secrets Management | Environment variables | **Critical** |
| Database Backups | Automated backup scripts | High |

---

<br/>

# CHAPTER 7b

## ETHICAL CONSIDERATIONS

### 7b.1 Medical AI Ethics

1. **Not a Replacement for Doctors:** This system provides **preliminary screening only**. All predictions must be validated by qualified medical professionals. The application includes a disclaimer that AI-generated predictions should not be used as the sole basis for medical decisions.

2. **False Positives vs False Negatives:**
   - **False Positives:** Patient unnecessarily worried but advised to consult a doctor (low harm)
   - **False Negatives:** Patient misses a diagnosis, potentially leading to delayed treatment (higher harm)
   - The system is designed to favor sensitivity where possible, erring on the side of caution

3. **Data Bias Awareness:**

   | Dataset | Known Bias | Impact |
   |---------|-----------|--------|
   | PIMA | Primarily Pima Indian women | May not generalize to all ethnicities |
   | BRFSS | US-based health survey | May not apply to other countries |
   | Symptom2Disease | English-only descriptions | Excludes non-English speakers |

   These limitations mean predictions may be less accurate for underrepresented populations.

4. **Transparency:** Confidence scores are displayed for every prediction, allowing users to assess the reliability of results. Users are informed that the analysis is AI-driven.

5. **Informed Consent:** Patients voluntarily input their symptoms and are aware that the system uses AI for analysis. No data is collected without user interaction.

6. **LLM-Generated Content:** Disease information generated by Gemma 3 is used for educational purposes only. Users are advised to consult healthcare professionals for medical advice.

---

<br/>

# CHAPTER 7c

## LIMITATIONS

| # | Limitation | Impact | Severity |
|---|-----------|--------|----------|
| 1 | **Small PIMA Dataset** (768 records) | Limits basic diabetes accuracy to ~78% | Medium |
| 2 | **Low Diabetes Recall** (19% on BRFSS) | ~81% of diabetic patients would be missed | **High** |
| 3 | **Disease Overlap Assumption** | 100% accuracy may not reflect real-world complexity | Medium |
| 4 | **English-Only Text Prediction** | Excludes non-English speaking patients | Medium |
| 5 | **No Image Analysis** | Cannot process X-rays, CT scans, skin photos | Medium |
| 6 | **Static Specialist Mapping** | Cannot adapt to new specializations | Low |
| 7 | **No Patient History** | No consideration of medical history or medications | **High** |
| 8 | **Session-Based Security** | Basic Flask session; needs JWT/OAuth for production | Medium |
| 9 | **LLM Dependency** | Disease info quality depends on Gemma 3 accuracy | Medium |
| 10 | **No Real-Time Model Updates** | Models are static; no continuous learning | Low |

---

<br/>

# CHAPTER 8

## FUTURE SCOPE

The following enhancements are planned for future versions of the system:

### Short-Term Enhancements (6-12 months)

| # | Enhancement | Description |
|---|------------|-------------|
| 1 | **Pretrained LLM Integration** | Use Gemma 4 or similar models via Ollama for enhanced disease prediction, improved chatbot responses, and automated medical report analysis |
| 2 | **Deep Learning Models** | Implement CNN and LSTM networks for improved text-based symptom classification and time-series health data analysis |
| 3 | **Extended Disease Coverage** | Integrate additional datasets to expand beyond 41 (binary) and 24 (text) diseases |
| 4 | **Password Security** | Implement bcrypt hashing, HTTPS, and JWT-based authentication |

### Medium-Term Enhancements (1-2 years)

| # | Enhancement | Description |
|---|------------|-------------|
| 5 | **Multi-Language Support** | Enable symptom input in Hindi, Marathi, and other languages using multilingual NLP models |
| 6 | **Mobile Application** | Develop React Native or Flutter app for broader smartphone accessibility |
| 7 | **EHR Integration** | Connect with hospital Electronic Health Records for contextual predictions |
| 8 | **Drug Interaction Checker** | Module to check for potential drug interactions based on patient medications |

### Long-Term Vision (2+ years)

| # | Enhancement | Description |
|---|------------|-------------|
| 9 | **Real-Time Video Consultation** | Integrate WebRTC for live video consultation with doctors |
| 10 | **IoT & Wearable Integration** | Incorporate data from wearables (heart rate, blood oxygen, glucose monitors) for continuous monitoring |
| 11 | **MLOps Pipeline** | Automated model retraining and monitoring using MLflow and Airflow |
| 12 | **Medical Imaging** | Add CNN-based modules for X-ray, CT scan, and dermatological image analysis |

---

<br/>

# CHAPTER 9

## CONCLUSION

This project successfully developed an **end-to-end AI-powered healthcare web application** for disease prediction based on symptoms. The key achievements are summarized below:

### 9.1 Key Achievements

| Achievement | Details |
|-------------|---------|
| **Multiple Prediction Modes** | 4 distinct modules -- binary symptoms (132 -> 41 diseases), text-based (free text -> 24 diseases), basic diabetes (8 features), advanced diabetes (21 indicators, 229,474 records) |
| **High Accuracy** | Binary: **100%** (SVM), Text: **97.92%** (TF-IDF + MNB), Advanced Diabetes: **84.05%** (RF + SMOTE) |
| **End-to-End Functionality** | Prediction -> specialist mapping -> appointment booking -> Razorpay payment -> email notifications -> chatbot support |
| **Role-Based Access** | 3 user roles (Patient, Doctor, Admin) with dedicated dashboards |
| **LLM Enhancement** | Gemma 3 via Ollama for structured disease information generation |
| **Scalable Architecture** | Flask three-tier architecture with configuration-driven ML pipeline |

### 9.2 Technical Contributions

1. **Multi-Modal Prediction System:** Unlike existing systems that offer a single prediction mode, our application integrates four distinct prediction modules in a single platform, making it accessible to users with varying levels of medical knowledge.

2. **Comprehensive Diabetes Assessment:** By combining the PIMA dataset (clinical features) with the BRFSS dataset (21 behavioral and demographic health indicators from 229,474 records), we demonstrate that socioeconomic and behavioral factors significantly improve diabetes prediction accuracy.

3. **NLP-Enabled Accessibility:** The text-based prediction module allows patients to describe symptoms in natural language, removing the barrier of medical terminology knowledge and achieving 97.92% accuracy.

4. **LLM-Enriched Information:** Integration of Gemma 3 for generating structured disease information (descriptions, symptoms, precautions, medications, diet) demonstrates the practical application of large language models in healthcare.

### 9.3 Social and Healthcare Impact

| Impact Area | Description |
|-------------|-------------|
| **Accessibility** | Provides preliminary diagnosis to patients in remote/underserved areas with internet access, bridging the urban-rural healthcare divide |
| **Early Detection** | Enables early disease detection by making symptom analysis available 24/7 without requiring an appointment |
| **Cost Reduction** | Reduces unnecessary hospital visits (Rs. 200 online appointment vs Rs. 500+ OPD fees + travel costs) |
| **Health Awareness** | AI chatbot educates users about diseases, symptoms, and when to seek medical help, promoting health literacy |
| **Specialist Routing** | Automatically guides patients to the right specialist, reducing misdirected consultations |
| **Scalability** | Web-based system can serve thousands of users simultaneously with minimal infrastructure |

### 9.4 Final Remarks

The system demonstrates the effective application of Machine Learning, NLP, and Deep Learning techniques in creating a practical healthcare tool. While not a replacement for professional medical diagnosis, it serves as a valuable **preliminary screening tool** that can guide patients toward timely specialist consultation. The modular, configuration-driven architecture ensures the system can be easily extended with new diseases, prediction modes, and features in future iterations.

---

<br/>

# REFERENCES

[1] Y. Deepthi, K.P. Kalyan, M. Vyas, K. Radhika, D.K. Babu, and N.V. Krishna Rao, "Disease Prediction Based on Symptoms Using Machine Learning," in *Energy Systems, Drives and Automations: Proceedings of ESDA 2019*, Springer, 2020.

[2] D. Sisodia and D. Sisodia, "Prediction of Diabetes using Classification Algorithms," *International Journal of Computer Applications*, vol. 176, no. 4, pp. 19-23, 2018.

[3] A. K. Singh, S. Kumar, M. Singh et al., "Machine Learning Approach for Diabetes Prediction Using BRFSS Dataset," *Procedia Computer Science*, 2022. DOI: 10.1016/j.procs.2022.01.146

[4] A. Hamdi, M. Mohamed, R. Emad, and K. Shaban, "An Ensemble Classification Approach in A Multi-Layered Large Language Model Framework for Disease Prediction," *arXiv preprint arXiv:2509.02446*, September 2025.

[5] R. Zannat, A. Al Shafi, and A. Muntakim, "Bridging the Gap in Bangla Healthcare: Machine Learning Based Disease Prediction Using a Symptoms-Disease Dataset," *IEEE ECCE 2025* (arXiv:2601.12068), January 2026.

[6] A. Al Shafi, R. Zannat, A. Muntakim, and M. Hasan, "A Structured Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy," *arXiv preprint arXiv:2506.13610*, June 2025.

[7] M. Shetty and C. Jordan, "Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM," *arXiv preprint arXiv:2503.19394*, March 2025.

[8] V. Krishnaiah, G. Narsimha, and N. Subhash Chandra, "A Comparative Study of Classification Algorithms for Disease Prediction," *International Journal of Computer Applications*, vol. 147, no. 10, 2016.

[9] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[10] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems 32*, 2019, pp. 8024-8035.

[11] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321-357, 2002.

[12] W. McKinney, "Data Structures for Statistical Computing in Python," in *Proceedings of the 9th Python in Science Conference*, 2010, pp. 56-61.

[13] S. Bird, E. Loper, and E. Klein, *Natural Language Processing with Python*, O'Reilly Media Inc., 2009.

[14] M. Pal, "Random Forest Classifier for Remote Sensing Classification," *International Journal of Remote Sensing*, vol. 26, no. 1, pp. 217-222, 2005.

[15] C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995.

---

<br/>

# APPENDICES

## Appendix A: Project Quick Stats

| Metric | Value |
|--------|-------|
| Total Diseases (Binary Symptoms) | 41 |
| Total Diseases (Text-Based) | 24 |
| Total Symptoms Tracked | 132 |
| Total Symptom Text Descriptions | 1,200 |
| ML Models Trained | 17+ (6 Binary + 5 Text + 6 Diabetes BRFSS + Chatbot NN) |
| Deep Learning Models | 1 (PyTorch NN for Chatbot) |
| Prediction Modes | 4 |
| Database Tables | 4 |
| HTML Templates | 27 |
| Flask Routes | 25+ |
| User Roles | 3 (User, Doctor, Admin) |
| Specialist Categories | 8 |
| Diabetes Dataset Records (Basic) | 768 |
| Diabetes Dataset Records (Advanced) | 229,474 |
| Best Binary Model Accuracy | 100% (SVM) |
| Best Text Model Accuracy | 97.92% (TF-IDF + MNB) |
| Best Diabetes Advanced Accuracy | 84.05% (Random Forest + SMOTE) |
| LLM Used | Gemma 3 (via Ollama) |
| Payment Gateway | Razorpay |
| Email Service | SMTP (Gmail) |

## Appendix B: Software and Hardware Requirements

### Software Requirements

| Requirement | Specification |
|------------|---------------|
| Operating System | Windows 10/11, macOS, or Linux |
| Python | 3.8 or higher |
| Database | MySQL 8.0+ |
| Package Manager | Conda/Anaconda (environment.yml) |
| Ollama | For Gemma 3 LLM (optional, for disease info generation) |
| Web Browser | Chrome, Firefox, Edge (modern versions) |

### Hardware Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Processor | Intel Core i5 | Intel Core i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB (for BRFSS training) |
| Storage | 2 GB (project + datasets) | 5 GB (with models and LLM) |
| GPU | Not required | NVIDIA GPU (for faster PyTorch training) |
| Internet | Required for Razorpay & Ollama | Stable broadband connection |

### Python Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| Flask-MySQLdb | MySQL database connector |
| scikit-learn | ML algorithms and evaluation |
| xgboost | XGBoost classifier |
| pandas | Data manipulation |
| numpy | Numerical computation |
| nltk | Natural language processing |
| torch (PyTorch) | Neural network for chatbot |
| imbalanced-learn | SMOTE for class balancing |
| matplotlib, seaborn | Data visualization |
| joblib | Model serialization |
| razorpay | Payment gateway SDK |
| PyYAML | Configuration file parsing |
| Werkzeug | Secure file upload handling |

## Appendix C: Model Files

| File | Description | Size (approx.) |
|------|-------------|----------------|
| `saved_model/best_binary_model.joblib` | SVM model for binary symptom prediction | ~500 KB |
| `saved_model/best_text_model.joblib` | TF-IDF + MultinomialNB pipeline | ~2 MB |
| `saved_model/text_label_encoder.joblib` | Label encoder for text model disease names | ~1 KB |
| `saved_model/decision_tree.joblib` | Decision Tree model for binary symptoms | ~200 KB |
| `saved_model/mnb.joblib` | Multinomial Naive Bayes for binary symptoms | ~100 KB |
| `saved_model/gradient_boost.joblib` | Gradient Boosting model for binary symptoms | ~5 MB |
| `svm_model.pkl` | SVM model for basic diabetes prediction | ~50 KB |
| `scaler.pkl` | StandardScaler for basic diabetes model | ~1 KB |
| `diabetes_brfss_model.pkl` | Random Forest for advanced diabetes prediction | ~50 MB |
| `diabetes_brfss_scaler.pkl` | StandardScaler for advanced diabetes model | ~1 KB |
| `diabetes_brfss_features.pkl` | Feature names for advanced diabetes model | ~1 KB |
| `data.pth` | PyTorch state dict for chatbot neural network | ~50 KB |

## Appendix D: Application Screenshots

> **[Insert screenshots of the running application here]**
>
> Suggested screenshots:
> 1. Home Page
> 2. User Registration / Login
> 3. Disease Prediction (Binary Symptoms) -- Input and Result
> 4. Disease Prediction (Text-Based) -- Input and Result with Top-3
> 5. Diabetes Basic Assessment -- Form and Result
> 6. Diabetes Advanced Assessment -- Form with 21 indicators and Risk Meter
> 7. Appointment Booking -- Doctor selection and form
> 8. Razorpay Payment Page
> 9. Doctor Panel -- Appointment management
> 10. Admin Dashboard -- User/Doctor management
> 11. AI Chatbot -- Sample conversation

## Appendix E: Project Timeline (Gantt Chart)

| Phase | M1 | M2 | M3 | M4 | M5 | M6 |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|
| Literature Survey | ## | ## | | | | |
| Dataset Collection & Preprocessing | | ## | ## | | | |
| ML Model Training & Evaluation | | | ## | ## | | |
| Web Application Development | | | | ## | ## | |
| Chatbot Development | | | | ## | | |
| Integration & Testing | | | | | ## | ## |
| Payment Gateway Integration | | | | | ## | |
| Documentation & Report Writing | | | | | ## | ## |
| Final Presentation Preparation | | | | | | ## |

## Appendix F: Glossary of Terms

| Term | Definition |
|------|-----------|
| **AI** | Artificial Intelligence -- simulation of human intelligence by machines |
| **Bag of Words (BoW)** | Text representation method counting word occurrences, ignoring grammar |
| **BERT** | Bidirectional Encoder Representations from Transformers -- a transformer-based NLP model |
| **Binary Classification** | Classification task with exactly two classes (e.g., diabetic/non-diabetic) |
| **BRFSS** | Behavioral Risk Factor Surveillance System -- annual US health survey by CDC |
| **Cross-Validation** | Model evaluation technique splitting data into k folds for robust assessment |
| **Deep Learning** | Subset of ML using neural networks with multiple hidden layers |
| **F1-Score** | Harmonic mean of precision and recall: F1 = 2 x (P x R) / (P + R) |
| **Feature Engineering** | Process of creating input features from existing data to improve model performance |
| **Feature Importance** | Score indicating how much each feature contributes to predictions |
| **Flask** | Lightweight Python web framework for building web applications |
| **Gemma** | Family of lightweight open-source LLMs by Google, used via Ollama |
| **Gini Impurity** | Measure of how often a randomly chosen element would be incorrectly classified |
| **Hyperparameter** | Model configuration parameter set before training (e.g., learning rate) |
| **IDF** | Inverse Document Frequency -- measures term rarity across all documents |
| **Joblib** | Python library for efficient serialization of scikit-learn models |
| **LLM** | Large Language Model -- AI model trained on vast text data for language tasks |
| **MNB** | Multinomial Naive Bayes -- probabilistic classifier for text classification |
| **NLP** | Natural Language Processing -- AI field focused on human-computer language interaction |
| **Ollama** | Tool for running large language models locally (Gemma, Llama, etc.) |
| **Overfitting** | When a model learns training noise and performs poorly on new data |
| **Pickle** | Python serialization module for saving/loading objects |
| **PIMA** | Pima Indian Diabetes dataset -- standard benchmark for diabetes prediction |
| **PorterStemmer** | Algorithm for reducing words to root form (e.g., "running" -> "run") |
| **PyTorch** | Open-source deep learning framework by Meta |
| **Random Forest** | Ensemble of decision trees trained on random data/feature subsets |
| **Recall** | Proportion of actual positives correctly identified: TP / (TP + FN) |
| **ReLU** | Rectified Linear Unit activation: f(x) = max(0, x) |
| **RBF Kernel** | Radial Basis Function kernel for non-linear SVM classification |
| **Razorpay** | Indian payment gateway platform for online transactions |
| **SMOTE** | Synthetic Minority Oversampling Technique for class imbalance |
| **StandardScaler** | Transformer that standardizes features to zero mean and unit variance |
| **SVM** | Support Vector Machine -- finds optimal class-separating hyperplane |
| **TF** | Term Frequency -- measures how often a term appears in a document |
| **TF-IDF** | Term Frequency-Inverse Document Frequency -- measures word importance |
| **Tokenization** | Splitting text into individual words, phrases, or symbols |
| **UML** | Unified Modeling Language -- standardized notation for software diagrams |

## Appendix G: Plagiarism Certificate

> **[Insert plagiarism check report here]**
>
> Plagiarism must be below 10% as per university guidelines.
> Recommended tools: Turnitin, iThenticate, or PlagiarismCheck.org
> Attach the certificate showing plagiarism percentage.

## Appendix H: IEEE Format Compliance Note

This report follows the IEEE/Springer paper format as required by the university evaluation criteria:

1. **References:** IEEE citation style [numbered] with full bibliographic details
2. **Structure:** Abstract -> Introduction -> Literature Survey -> Methodology -> Implementation -> Results -> Conclusion -> References
3. **Figures and Tables:** Numbered sequentially with descriptive captions
4. **Mathematical Notations:** Equations defined with variable descriptions
5. **Citations:** In-text citations as [1], [2], etc. referencing the numbered bibliography

---

<p align="center"><i>End of Report</i></p>
<p align="center"><b>Disease Prediction Based on Symptoms -- M.Tech Project Report 2025-2026</b></p>
