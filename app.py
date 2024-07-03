import numpy as np 
import pandas as pd 
import pickle as pk 
import streamlit as st 
import sklearn
from sklearn.metrics import accuracy_score

model = pk.load(open('model/heart_disease_model.pkl', 'rb')) 
data = pd.read_csv("data/heart_disease.csv")

st.header('**Heart Disease Prediction System**')

gender = st.selectbox('**Select Gender**', data['Gender'].unique())
if gender == 'Male':
    gen = 1
else:
    gen = 0

age = st.number_input("**Enter Patient Age**")

currentSmoker = st.number_input("**Enter Patient Current Smoker Status; 1 if yes , 0 if no**")

cigsPerDay = st.number_input("**Enter Number of Patient cigsPerDay**")

BPMeds = st.number_input("**Enter Patient BPMeds Status; 1 if yes , 0 if no**")

prevalentStroke = st.selectbox('**Is patient has prevalentStroke**', data['prevalentStroke'].unique())
if prevalentStroke == 'yes':
    prevalentStroke = 1
else:
    prevalentStroke = 0

prevalentHyp = st.number_input("**Enter Patient prevalentHyp Status; 1 if yes , 0 if no**")
diabetes = st.number_input("**Enter Patient Diabetes Status; 1 if yes , 0 if no**")
totChol = st.number_input("**Enter Patient totChol Status**")
sysBP = st.number_input("**Enter Patient sysBP Status**")
diaBP = st.number_input("**Enter Patient diaBP Status**")
BMI = st.number_input("**Enter Patient BMI Status**")
heartRate = st.number_input("**Enter Patient HeartRate Status**")
glucose = st.number_input("**Enter Patient Glucose Status**")

if st.button('**Click to Predict**'):
    input = np.array([[gen,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,
                                                           totChol,sysBP,diaBP,BMI,heartRate,glucose]])
    output = model.predict(input)
    if output[0] == 0:
        stn = '**Patient is Healthy, No Heart Disease**'
    else:
        stn = '**Patient May have Heart Disease**'
    st.markdown(stn)

