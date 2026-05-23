# ==========================================================
# 🩺 AI SYMPTOM CHECKER (STREAMLIT APP)
# OUTPUT: REAL DISEASE NAME (NOT CODE)
# ==========================================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Symptom Checker")
st.write("Predict possible disease based on symptoms")


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = pd.read_csv("ai_symptom_checker_dataset.csv")


# ==========================================================
# 3. SELECT FEATURES
# ==========================================================

features = [
    "Age",
    "Gender",
    "Body_Temperature_C",
    "Heart_Rate_bpm",
    "Oxygen_Saturation_%",
    "Blood_Sugar_mg_dL",
    "Pain_Level"
]

target = "Target_Disease"


# ==========================================================
# 4. SIMPLE ENCODING (BEGINNER FRIENDLY)
# ==========================================================

df_encoded = df.copy()

df_encoded["Gender"] = df_encoded["Gender"].map({
    "Male": 0,
    "Female": 1
})


# ==========================================================
# 5. SPLIT DATA
# ==========================================================

X = df_encoded[features]
y = df[target]   # KEEP ORIGINAL LABELS (IMPORTANT FIX)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# 6. TRAIN MODEL
# ==========================================================

model = RandomForestClassifier()
model.fit(X_train, y_train)


# ==========================================================
# 7. MODEL ACCURACY
# ==========================================================

accuracy = model.score(X_test, y_test)

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")


# ==========================================================
# 8. USER INPUT SECTION
# ==========================================================

st.header("Patient Information")

age = st.number_input("Age", 1, 100, 25)

gender = st.selectbox("Gender", ["Male", "Female"])

temperature = st.number_input("Body Temperature (°C)", 30.0, 45.0, 37.0)

heart_rate = st.number_input("Heart Rate", 40, 200, 80)

oxygen = st.number_input("Oxygen Level (%)", 50, 100, 98)

blood_sugar = st.number_input("Blood Sugar Level", 50.0, 400.0, 100.0)

pain = st.slider("Pain Level", 1, 10, 3)


# ==========================================================
# 9. PREDICTION
# ==========================================================

if st.button("Predict Disease"):

    # Encode gender
    gender_encoded = 0 if gender == "Male" else 1

    # Create input dataframe
    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        temperature,
        heart_rate,
        oxygen,
        blood_sugar,
        pain
    ]], columns=features)


    # Predict disease (THIS RETURNS REAL NAME)
    prediction = model.predict(input_data)[0]


    # ======================================================
    # OUTPUT RESULT
    # ======================================================

    st.subheader("Prediction Result")
    st.success(f"🧬 Possible Disease: {prediction}")


    # ======================================================
    # SIMPLE MEDICAL ADVICE
    # ======================================================

    st.subheader("Recommendation")

    if prediction == "Malaria":
        st.warning("Take malaria test and see a doctor.")

    elif prediction == "COVID-19":
        st.warning("Isolate and seek medical care.")

    elif prediction == "Pneumonia":
        st.error("Go to hospital immediately.")

    elif prediction == "Asthma":
        st.info("Use inhaler and monitor breathing.")

    elif prediction == "Flu":
        st.info("Rest and drink fluids.")

    else:
        st.info("Consult a medical professional.")


# ==========================================================
# FOOTER
# ==========================================================

st.write("---")
st.caption("AI Symptom Checker | Streamlit + Machine Learning")