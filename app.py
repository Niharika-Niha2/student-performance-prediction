import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(page_title="Student Result Predictor", page_icon="🎓", layout="centered")

# Custom CSS for pastel theme
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
        }
        h1 {
            color: #00695c;
        }
        .stButton>button {
            background-color: #4db6ac;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
with open('student_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App Title
st.title("🎓 Student Pass/Fail Prediction App")

# Optional: Student Info
st.text_input("👤 Student Name", key="name")
st.text_input("🆔 Roll Number", key="roll")

# Input Form
with st.form("input_form"):
    st.subheader("📥 Enter Student Details")

    study_hours = st.slider("📚 Study Hours", 0, 10, 4)
    attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
    internal_marks = st.slider("📝 Internal Marks", 0, 100, 60)
    sleep_hours = st.slider("💤 Sleep Hours", 0, 10, 6)
    extracurricular = st.radio("🏃‍♀️ Extracurricular Activities", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

    submit = st.form_submit_button("🔮 Predict")

if submit:
    # Create DataFrame from inputs
    input_df = pd.DataFrame([[study_hours, attendance, internal_marks, sleep_hours, extracurricular]],
                            columns=['study_hours', 'attendance', 'internal_marks', 'sleep_hours', 'extracurricular'])

    # Make prediction
    prediction = model.predict(input_df)
    result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"

    # Output
    st.success(f"🎯 Prediction Result: **{result}**")

    st.markdown("📊 **Input Summary**")
    st.table(input_df)
