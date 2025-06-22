import streamlit as st
import pandas as pd
import joblib

# Load the trained model (no scaler needed)
model = joblib.load('logistic_model.pkl')

# App title and description
st.title("Dry Eye Disease Prediction App")
st.write("Enter your health information to predict dry eye disease risk")

# Create two columns for input
col1, col2 = st.columns(2)

# Column 1: Personal Information
with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    st.subheader("Lifestyle & Health")
    sleep_duration = st.number_input("Sleep Duration (hrs)", min_value=1.0, max_value=12.0, value=7.0)
    sleep_quality = st.slider("Sleep Quality (1-5)", min_value=1, max_value=5, value=4)
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)

# Additional health metrics
st.subheader("Health Metrics")
col3, col4 = st.columns(2)

with col3:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

with col4:
    sleep_disorder = st.selectbox("Sleep Disorder", ["Yes", "No"])
    caffeine_consumption = st.selectbox("Caffeine Consumption", ["Yes", "No"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    blue_light_filter = st.selectbox("Blue-light Filter", ["Yes", "No"])

# Screen time
screen_time = st.slider("Average Screen Time (hrs/day)", min_value=0.0, max_value=16.0, value=6.0)

# Prediction button
if st.button("Predict Dry Eye Disease Risk"):
    try:
        # Prepare input data - encode categorical variables
        gender_encoded = 1 if gender == "M" else 2  # M=1, F=2
        sleep_disorder_encoded = 1 if sleep_disorder == "Yes" else 0
        caffeine_encoded = 1 if caffeine_consumption == "Yes" else 0
        smoking_encoded = 1 if smoking == "Yes" else 0
        blue_light_encoded = 1 if blue_light_filter == "Yes" else 0
          # Create feature array in the exact order expected by the model
        # Based on notebook: Gender, Age, Sleep Duration, Sleep Quality, Stress Level, 
        # Blood pressure, Heart Rate, Daily Steps, Weight, Sleep Disorder, 
        # Caffeine Consumption, Smoking, Average Screen Time, Blue-light Filter
        features = [
            gender_encoded,
            age,
            sleep_duration,
            sleep_quality,
            stress_level,
            heart_rate,
            daily_steps,
            weight,
            sleep_disorder_encoded,
            caffeine_encoded,
            smoking_encoded,
            screen_time,
            blue_light_encoded
        ]
          # Convert to DataFrame for prediction
        feature_names = [
            'Gender', 'Age', 'Sleep duration', 'Sleep quality', 'Stress level',
            'Heart rate', 'Daily steps', 'Weight', 'Sleep disorder',
            'Caffeine consumption', 'Smoking', 'Average screen time', 'Blue-light filter'
        ]
        
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Debug: Show the input values
        st.write("Debug - Input values:")
        st.write(input_df)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("⚠️ Dry Eye Disease")
        else:
            st.success("✅ No Dry Eye Disease")
        

        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that the model file exists and was trained with the correct features.")
