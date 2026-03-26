#  Student Performance Indicator (ML Project)

##  Project Overview
This project predicts a student’s **writing score** based on various factors such as:

- Gender  
- Race/Ethnicity  
- Parental Level of Education  
- Lunch Type  
- Test Preparation Course  
- Math Score  
- Reading Score  

The goal is to build a **machine learning pipeline** that processes data, trains multiple models, and selects the best-performing one.

---

##  Project Workflow

The pipeline follows these steps:

### 1. Data Ingestion
- Reads raw dataset  
- Splits into train and test sets  
- Saves datasets in `artifacts/`  

### 2. Data Transformation
- Handles missing values  
- Encodes categorical features (OneHotEncoding)  
- Scales numerical features (StandardScaler)  
- Saves preprocessing object  

### 3. Model Training & Evaluation
- Trains multiple regression models  
- Uses **RandomizedSearchCV** for hyperparameter tuning  
- Evaluates using **R² score**  
- Selects and saves the best model  

### 4. Reporting
- Displays train and test performance of all models  
- Highlights the best model  

---

##  Models Used

- Random Forest Regressor  
- Decision Tree Regressor  
- Gradient Boosting Regressor  
- Linear Regression  
- K-Nearest Neighbours  
- XGBoost Regressor  
- CatBoost Regressor  
- AdaBoost Regressor  
- Support Vector Regressor  

---

##  Model Performance

