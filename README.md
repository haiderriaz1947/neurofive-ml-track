# 🚀 NeuroFive ML Track

This repository contains my work from the **NeuroFive Machine Learning Track**, where I am building my Machine Learning skills through practical, real-world projects.

The journey started with understanding data, moved into predictive modeling, and progressed toward solving real **business and environmental problems with Machine Learning**, building professional ML pipelines, handling messy data, and deploying models as usable applications.

---

# 📊 Week 1: Titanic — Exploratory Data Analysis

## 🔍 What I Did

- Loaded and explored the Titanic dataset using **Pandas**
- Used `.head()`, `.info()` and `.describe()` for data understanding
- Identified missing values
- Classified categorical and numerical features
- Cleaned and prepared the dataset
- Created visualizations using **Matplotlib & Seaborn**
- Explored relationships between passenger characteristics and survival

## 🧠 Key Insights

- 👩 **Gender had the biggest impact on survival**
- 🎫 **Passenger class strongly influenced survival**
- 💰 Higher fares were associated with better survival chances
- 👶 Age had a comparatively smaller impact
- 👨‍👩‍👧 Family presence showed a smaller effect on survival

### 📌 Key Learning

> **Before building a model, understand the data first.**

---

# 🤖 Week 2: Titanic Survival Prediction — Logistic Regression

## ⚙️ What I Did

- Prepared the cleaned Titanic dataset
- Handled missing values
- Encoded categorical variables
- Split data into training and testing sets
- Built a **Logistic Regression** classification model
- Predicted passenger survival
- Evaluated the model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

## 🧠 Key Learning

This project introduced my first complete Machine Learning workflow:

**Data Cleaning → Feature Preparation → Train/Test Split → Model Training → Prediction → Evaluation**

It helped me understand how a model can learn patterns from historical data and use them to make predictions on unseen data.

---

# 📞 Week 3: Customer Churn Prediction — Business Problem

## 💼 Problem

Customer churn is a real business challenge. Companies need to identify customers who may leave so they can take action before losing them.

For this project, I worked with the **Telco Customer Churn** dataset.

## 🔍 What I Did

- Performed quick Exploratory Data Analysis
- Analyzed churn patterns across:
  - Contract Type
  - Tenure
  - Monthly Charges
- Checked missing values
- Identified class imbalance
- Handled categorical variables using **OneHotEncoder**
- Built a **Decision Tree Classifier**
- Built a **Logistic Regression model**
- Compared both models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Created confusion matrices
- Identified the **Top 3 churn-driving features** using `feature_importances_`

## 💡 Business Perspective

The project goes beyond simply predicting churn. It explores which customer characteristics provide the strongest signals for identifying customers at risk of leaving.

The findings can help businesses think about:

- 🎯 Targeted retention campaigns
- 📞 Customer engagement strategies
- 💰 Pricing and service decisions
- 🔄 Customer retention planning

### 📌 Key Learning

> **Machine Learning becomes more valuable when predictions can be connected to a real business decision.**

---

# 🛠️ Week 4: ML Pipelines & Ensemble Learning

Week 4 focused on moving from basic notebook-based modeling toward **cleaner, reusable and more professional Machine Learning workflows**.

---

## 🔧 Task 1: ML Pipeline with Feature Engineering

Built a proper **Scikit-learn Pipeline** using `ColumnTransformer` to combine preprocessing and modeling into a single reusable workflow.

### 🔍 What I Did

- Applied **StandardScaler** to numerical features
- Applied **OneHotEncoder** to categorical features
- Combined preprocessing and the ML model into one Pipeline
- Created **2+ engineered features**
- Tested whether feature engineering improved model performance
- Evaluated the pipeline against the earlier manual approach
- Saved the final trained pipeline using **Joblib**

📓 **Notebook:**

[Churn Pipeline & Feature Engineering](https://github.com/haiderriaz1947/neurofive-ml-track/blob/main/churn_pipeline_feature_engineering.ipynb)

### 📌 Key Learning

> **A professional ML workflow should be reproducible, reusable and protected against inconsistent preprocessing and data leakage.**

---

## 🌲 Task 2: Ensemble Learning — Random Forest vs XGBoost

The second Week 4 task explored **ensemble learning**, comparing powerful tree-based ensemble methods with earlier single-model approaches.

### 🔍 What I Did

- Trained a **Random Forest Classifier**
- Trained an **XGBoost Classifier**
- Compared ensemble models with earlier Logistic Regression / Decision Tree models
- Analyzed and plotted **feature importances**
- Compared which features were considered important by each model
- Created a model comparison table
- Explored how ensemble methods improve prediction performance

### 🧠 Random Forest vs XGBoost

**Random Forest** builds many decision trees independently using randomness in samples and features, then combines their predictions.

**XGBoost** builds trees sequentially, where each new tree focuses on correcting the errors made by previous trees.

Random Forest mainly reduces variance through averaging, while XGBoost uses boosting to progressively improve the model.

📓 **Notebook:**

[Random Forest vs XGBoost](https://github.com/haiderriaz1947/neurofive-ml-track/blob/main/Ensemble%20Learning%20-%20Random%20Forest%20vs%20XGBoost.ipynb)

### 📊 Model Comparison

| Model | Type | Evaluation |
|---|---|---|
| Logistic Regression | Single Model | Baseline |
| Decision Tree | Single Model | Baseline |
| Random Forest | Ensemble | Compared |
| XGBoost | Ensemble | Compared |

---

# ⚖️ Week 5: Handling Imbalanced & Messy Real-World Data

Week 5 focused on an important real-world Machine Learning challenge: **imbalanced datasets**.

The goal was to understand why accuracy alone can be misleading and how models can be improved when target classes are unevenly distributed.

---

## 💳 Task 1: Credit Card Fraud Detection — Imbalanced Data

For this task, I worked on a **Credit Card Fraud Detection** problem and investigated the imbalance between fraud and non-fraud transactions.

### 🔍 What I Did

- Checked and visualized the target class distribution
- Identified severe class imbalance
- Applied techniques for handling imbalanced data
- Retrained the model using a balanced approach
- Compared **Precision, Recall and F1-score** before and after handling imbalance
- Investigated why accuracy can be misleading when the minority class is rare

### 💡 Key Learning

> **In highly imbalanced classification problems, a high accuracy score does not necessarily mean the model is useful. Precision, Recall and F1-score provide a better view of minority-class performance.**

📓 **Notebook:**

[Fraud Detection — Imbalanced Data](https://github.com/haiderriaz1947/neurofive-ml-track/blob/main/Fraud%20Detection-Imbalanced.ipynb)

---

## 🌐 Task 2: Deploying a Machine Learning Model as a Web App

The next step was turning a trained ML model into something users can actually interact with.

I developed a **Streamlit web application** for the Customer Churn prediction model.

### 🚀 What I Did

- Saved the trained ML model using **Joblib**
- Built a user-friendly **Streamlit** interface
- Added input fields for customer features
- Added a **Predict** button
- Loaded the saved model inside the application
- Displayed the churn prediction to the user
- Deployed the application online

### 🔴 Live App

👉 [Open the NeuroFive Churn Prediction App](https://neurofive-ml-track-churn011.streamlit.app/)

### 📌 Key Learning

> **Building a model is only part of the ML journey — deploying it makes the model usable in a real-world setting.**

---

# 🌫️ Week 6: Capstone — End-to-End Machine Learning Project

Week 6 was the **capstone project**, where I applied the complete Machine Learning workflow to a real-world environmental problem: **predicting Lahore's next-day Air Quality Index (AQI).**

## 💡 Project: Lahore Next-Day AQI Prediction

### 🎯 Problem

Lahore regularly experiences poor air quality, particularly during the winter smog season. Poor AQI can affect outdoor activities, schools, hospitals, workplaces, and daily planning.

Most AQI information tells people about the air quality **today**, but there is value in knowing what conditions may look like **tomorrow**.

The goal of this project was to build an end-to-end Machine Learning solution capable of predicting **next-day AQI** using weather conditions, environmental factors, and recent AQI history.

---

## 📊 Dataset

The project uses a daily Lahore AQI dataset containing environmental and weather-related information.

### Main Features

- `date`
- `temperature_c`
- `humidity_pct`
- `wind_speed_kmh`
- `rainfall_mm`
- `crop_burning_index`
- `aqi`

Additional features were engineered during the Machine Learning workflow to capture seasonal patterns and recent AQI behavior.

---

## 🔍 What I Did

- Defined the AQI prediction problem
- Prepared and cleaned the dataset
- Checked missing values and data quality
- Performed Exploratory Data Analysis
- Analyzed AQI trends over time
- Investigated seasonal patterns
- Studied relationships between weather variables and AQI
- Created feature-engineering transformations
- Created seasonal features
- Created cyclical month features
- Created weekend and winter indicators
- Created historical AQI lag features
- Used a time-based train/test split
- Trained multiple regression models
- Evaluated model performance
- Selected the best-performing model
- Saved the final model using **Joblib**
- Built a **Streamlit web application**
- Deployed the application online

---

## 🧠 Feature Engineering

To improve the model's ability to learn temporal and environmental patterns, I created additional features such as:

- Month-based seasonal features
- Cyclical month encoding
- Weekend indicator
- Winter indicator
- Previous-day AQI
- AQI from 7 days ago
- Rolling 7-day AQI average

These features help the model capture both **seasonality** and **recent AQI behavior**.

---

## 🤖 Models Trained

I trained and compared multiple regression models:

| Model | Type |
|---|---|
| Linear Regression | Baseline Regression |
| Ridge Regression | Regularized Regression |
| Random Forest | Ensemble Model |
| Gradient Boosting | Ensemble Model |

### 📏 Evaluation Metrics

The models were evaluated using:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination

The best-performing model was selected based on the evaluation results and saved for deployment.

---

## 🔄 End-to-End ML Workflow

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Time-Based Train/Test Split
        ↓
Train Multiple Models
        ↓
Model Evaluation
        ↓
Select Best Model
        ↓
Save Model with Joblib
        ↓
Build Streamlit Application
        ↓
Deploy Application
