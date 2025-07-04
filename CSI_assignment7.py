# I had got a project for predicting student final exam marks based on study hours, attendance, and past scores. Hence, 
# I used joblib to save the best model and scaler from that ipynb file, & now I'll be using them in my streamlit app.

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_artifacts():
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

st.title("Student's Final Exam Score Predictor")
st.write("Enter study hours, attendance rate, and past exam score to predict final marks.")

col1, col2, col3 = st.columns(3)
with col1:
    study_hours = st.number_input("Study Hours per Week", value=10.0, min_value=0.0, max_value=39.0, step=0.5)
with col2:
    attendance_rate = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=80)
with col3:
    past_score = st.number_input("Past Exam Score (%)", value=70.0, min_value=50.0, max_value=100.0)

if st.button("Predict Final Marks"):

    features = np.array([[study_hours, attendance_rate, past_score]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.subheader("Predicted Final Exam Marks")
    st.success(f"{prediction:.2f}%")

    st.write("### Input Summary (Pie Chart)")
    labels = ["Study Hours", "Attendance Rate", "Past Score"]
    values = [study_hours, attendance_rate, past_score]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon'])
    ax.axis('equal')
    st.pyplot(fig)
