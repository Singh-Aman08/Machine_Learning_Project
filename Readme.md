#  Student Performance Indicator

A complete end-to-end Machine Learning project that predicts a student's **writing score** based on demographic and academic features.  
The project includes data ingestion, preprocessing, model training, evaluation, prediction pipeline, and a Streamlit web application.

---
##  About the Dataset

This project uses the **Students Performance in Exams Dataset** sourced from Kaggle:  
🔗 https://www.kaggle.com/datasets/spscientist/students-performance-in-exams  

The dataset contains **1000 student records**, capturing performance across three core academic subjects:
- **Mathematics**
- **Reading**
- **Writing**

In addition to exam scores, the dataset includes several **demographic and socio-economic attributes** that may influence student performance, such as:
- Gender  
- Race/Ethnicity  
- Parental level of education  
- Lunch type  
- Test preparation course completion  

These features provide valuable context for analyzing patterns in student achievement and understanding how different factors impact academic outcomes.

---

##  Project Overview

This project aims to build a robust ML pipeline to predict student performance using features such as:

- Gender  
- Race/Ethnicity  
- Parental Level of Education  
- Lunch Type  
- Test Preparation Course  
- Math Score  
- Reading Score  

The target variable is:
 **Writing Score**

---

##  Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **CatBoost, XGBoost**
- **Dill (for model serialization)**
- **Streamlit (for UI)**
- **Logging & Exception Handling**

---

##  ML Pipeline

The project follows a modular pipeline approach:

### Components:

- **Data Ingestion**
  - Reads dataset
  - Splits into train/test

- **Data Transformation**
  - Handles missing values
  - Encoding & scaling using `ColumnTransformer`

- **Model Training**
  - Trains multiple models:
    - Random Forest
    - Decision Tree
    - Gradient Boosting
    - Linear Regression
    - KNN
    - XGBoost
    - CatBoost
    - AdaBoost
    - SVM
  - Hyperparameter tuning using `RandomizedSearchCV`

- **Model Evaluation**
  - Uses **R² Score** for performance comparison

- **Prediction Pipeline**
  - Loads trained model & preprocessor
  - Transforms input data
  - Generates predictions

##  Model Performance

| Model                     | R2_Train | R2_Test |
|--------------------------|----------|---------|
| Random Forest            | 0.9646   | 0.9103  |
| Decision Tree            | 0.9279   | 0.8963  |
| Gradient Boosting        | 0.9935   | 0.9242  |
| Linear Regression        | 0.9499   | 0.9381  |
| K-Nearest Neighbours     | 0.9997   | 0.8954  |
| XGBoost                  | 0.9672   | 0.9308  |
| CatBoost                 | 0.9595   | 0.9251  |
| AdaBoost                 | 0.9288   | 0.9165  |
| Support Vector Machine   | 0.9496   | 0.9378  |

 **Best Model:** Linear Regression  
 **Test R² Score:** **0.9381**

---
## Project Files Description

### Artifacts (`artifacts/`)
- **model.pkl** – The trained machine learning model saved using `dill`.  
- **preprocessor.pkl** – Preprocessing pipeline (`ColumnTransformer`) used for encoding and scaling features.  
- **data.csv** – Original/raw dataset used for training and testing.  
- **train.csv** – Training dataset split from the original data.  
- **test.csv** – Test dataset split from the original data.  

### Source Code (`src/`)
- **__init__.py** – Marks the `src` folder as a Python package.  
- **data_ingestion.py** – Reads the raw dataset, splits it into train and test sets, and saves them to `artifacts/`.  
- **data_transformation.py** – Preprocesses data: handles missing values, encodes categorical features, scales numerical features, and returns a preprocessor object.  
- **model_trainer.py** – Trains multiple regression models, performs hyperparameter tuning using `RandomizedSearchCV`, evaluates models, and selects the best one.  
- **exception.py** – Custom exception class for consistent error handling across the project.  
- **rootlogger.py** – Logging setup to track workflow progress and errors.  
- **utils.py** – Utility functions such as saving objects (`dill`) and evaluating model performance.  

### Main Scripts
- **main.py** – Executes the full end-to-end machine learning pipeline: data ingestion → preprocessing → model training → evaluation.  
- **predict.py** – Defines the prediction pipeline with a `CustomData` class for user inputs and a `PredictionPipeline` class for generating predictions.  
- **app.py** – Streamlit web application for real-time prediction with interactive sliders for input features.  

### Others
- **setup.py** – Optional script for packaging and installing the project as a Python package.  
- **requirements.txt** – Lists all Python dependencies required to run the project.  
- **.gitignore** – Specifies files and folders to ignore in Git (e.g., artifacts, virtual environments).  
- **README.md** – This file: provides project overview, instructions, and documentation.

## Setup & Installation

1. **Clone the repository**

2. **Create Virtual Environment** -> conda create -n mlproject python=3.13 ->  conda activate mlproject  

3. **Install Dependencies** -> pip install -r requirements.txt  

## How to Run

1. **Run python main.py** : This will train models, evaluate performance, save model & preprocessor in artifacts/  

2. **Run python predict.py** : This will take the data and predict the result. 

3. **Run streamlit run app.py** : This will open in your browser, allowing you to start interacting with the Streamlit application. 
---
 
