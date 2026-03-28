import streamlit as st
import pandas as pd
from predict import CustomData, PredictionPipeline
from src.rootlogger import logger


st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's writing score based on their details and scores.")
st.sidebar.header("Enter Student Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ("Group A", "Group B", "Group C", "Group D", "Group E"))
    parental_level_of_education = st.sidebar.selectbox("Parental Level of Education", ("Intermediate","Undergraduate", "Postgraduate"))
    test_prep_course = st.sidebar.selectbox("Test Preparation Course", ("No Prep", "Completed Course"))
    lunch = st.sidebar.selectbox("Lunch", ("Free/Reduced Lunch", "Standard Lunch"))
    maths_score = st.sidebar.slider("Maths Score",0,100,70 )
    reading_score = st.sidebar.slider("Reading Score",0,100,70 )
    
    data = CustomData(gender, race_ethnicity, parental_level_of_education, lunch, test_prep_course, maths_score, reading_score)
    return data
    

input_data = user_input_features()
df = input_data.get_data_as_dataframe()

st.subheader("Entered Student Data")
st.write(df)

predict = PredictionPipeline()
try:
    pred = predict.predict(df)
    st.subheader("Predicted Writing Score")
    st.success(f"{pred[0]:.2f}")
    
    
    
except Exception as e:
    st.error("Prediction failed. Check logs for details.")
    logger.error(f"Prediction failed: {e}")
    
    





    
    
    