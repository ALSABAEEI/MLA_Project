import streamlit as st
import joblib
import pandas as pd

# Load trained Logistic Regression model
model = joblib.load("logistic_model_top5.pkl")
features = ['Gender', 'Sleep disorder', 'Caffeine consumption', 'Average screen time', 'Blue-light filter']

# App Title
st.set_page_config(page_title="Dry Eye Detection", layout="centered")
st.title("👁 Dry Eye Disease Detection")
st.write("This system uses a trained Logistic Regression model to predict the possibility of Dry Eye Disease based on lifestyle and screen habits.")

st.markdown("---")

# Input Form
with st.form("input_form"):
    st.subheader("📋 Basic & Lifestyle Information")

    
    age = st.number_input("🎂 Age", min_value=10, max_value=100, value=25)
    heart_rate = st.slider("❤️ Heart Rate (bpm)", 40, 120, 75)
    sleep_duration = st.slider("🛏️ Sleep Duration (hours)", 0.0, 12.0, step=0.5, value=6.0)
    sleep_quality = st.selectbox("😴 Sleep Quality", ['Poor', 'Fair', 'Good'])
    stress_level = st.slider("😣 Stress Level (1=Low, 10=High)", 1, 10, 5)
    blood_pressure = st.selectbox("🩸 Blood Pressure", ['Low', 'Normal', 'High'])
    daily_steps = st.number_input("🚶 Daily Steps", min_value=0, max_value=50000, value=5000)
    weight = st.number_input("⚖️ Weight (kg)", min_value=20.0, max_value=200.0, step=0.5, value=65.0)
    smoking = st.selectbox("🚬 Do you smoke?", ['No', 'Yes'])

    st.markdown("### ✅ Most important and effective information")
    gender = st.selectbox("👤 Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    sleep_disorder = st.selectbox("🛌 Sleep Disorder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    caffeine = st.selectbox("☕ Caffeine Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    screen_time = st.slider("📱 Average Screen Time (hours/day)", 0.0, 24.0, step=0.5, value=6.0)
    blue_light = st.selectbox("🔵 Blue-Light Filter Used?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    submitted = st.form_submit_button("🔍 Predict")

# Prediction Logic
if submitted:
    # Prepare input for model prediction
    user_data = pd.DataFrame([[gender, sleep_disorder, caffeine, screen_time, blue_light]], columns=features)
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][prediction]

    # Show result
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"🔴 Risk Detected: You have Dry Eye Disease.")
    else:
        st.success(f"🟢 No Dry Eye Detected.")

    # Show extra info
    st.markdown("---")
    st.subheader("📌 Additional Information (Not Used in Prediction)")
    st.markdown(f"""
    - **Age:** {age}  
    - **Heart Rate:** {heart_rate} bpm  
    - **Sleep Duration:** {sleep_duration} hours  
    - **Sleep Quality:** {sleep_quality}  
    - **Stress Level:** {stress_level}  
    - **Blood Pressure:** {blood_pressure}  
    - **Daily Steps:** {daily_steps}  
    - **Weight:** {weight} kg  
    - **Smoking:** {smoking}
    """)
