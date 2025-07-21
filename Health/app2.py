import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# Load the trained models
with open('diabetes_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('heart_model.pkl', 'rb') as file:
    heart_model = pickle.load(file)

def dual_input(label, slider_min, slider_max, default_val, step=1.0, is_float=False, help_text=""):
    col1, col2 = st.columns([3, 1])
    with col1:
        slider_value = st.slider(label, min_value=slider_min, max_value=slider_max, value=default_val,
                                 step=step, key=label + "_slider", help=help_text)
    with col2:
        input_type = st.number_input if not is_float else st.number_input
        typed_value = input_type(" ", value=slider_value, step=step, key=label + "_input", help=help_text)

    return typed_value

def plot_input_data(input_data, model_selection):
    st.subheader(f"{model_selection} - Input Data Overview")
    fig = px.bar(
        x=list(input_data.keys()),
        y=list(input_data.values()),
        labels={'x': 'Feature', 'y': 'Value'},
        color=list(input_data.values()),
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Health Prediction", layout="wide")
    st.title("🩺 Health Prediction App")

    st.sidebar.title("Model Selector")
    model_selection = st.sidebar.radio("Choose Prediction Model", ["Diabetes Prediction", "Heart Disease Prediction"])

    if model_selection == "Diabetes Prediction":
        st.header("🔬 Diabetes Prediction Inputs")

        with st.expander("ℹ️ Safe Ranges Reference"):
            st.markdown("""
            - **Glucose**: 70–99 mg/dL (fasting)
            - **Blood Pressure**: <120/80 mm Hg
            - **Cholesterol**: <200 mg/dL
            - **BMI**: 18.5–24.9 normal
            - **Insulin**: 2–25 μU/mL (fasting)
            - **Oldpeak**: ≤2.0 usually considered normal
            """)

        pregnancies = dual_input("Pregnancies", 0, 20, 2, step=1, help_text="Number of times pregnant. Normal varies per individual.")
        glucose = dual_input("Glucose Level (mg/dl)", 0, 300, 110, step=1, help_text="Normal fasting glucose: 70–99 mg/dL. ≥126 may indicate diabetes.")
        blood_pressure = dual_input("Blood Pressure (mm Hg)", 0, 200, 70, step=1, help_text="Normal systolic BP: <120 mm Hg. Hypertension: ≥130 mm Hg.")
        skin_thickness = dual_input("Skin Thickness (mm)", 0, 100, 20, step=1, help_text="Triceps skinfold thickness. Typical: ~20–35 mm.")
        insulin = dual_input("Insulin Level (μU/mL)", 0, 800, 80, step=1, help_text="Normal fasting insulin: 2–25 μU/mL. High may indicate insulin resistance.")
        bmi = dual_input("Body Mass Index (BMI)", 0.0, 70.0, 25.0, step=0.1, is_float=True, help_text="Healthy BMI: 18.5–24.9. Overweight: 25–29.9. Obesity: 30+.")
        diabetes_pedigree = dual_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01, is_float=True, help_text="Likelihood of diabetes based on family history. Higher = greater risk.")
        age = dual_input("Age", 0, 120, 30, step=1, help_text="Age in years. Risk increases with age, especially after 45.")

        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'Blood Pressure': blood_pressure,
            'Skin Thickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'Diabetes Pedigree': diabetes_pedigree,
            'Age': age
        }

        if st.button("Visualize Inputs"):
            plot_input_data(input_data, "Diabetes")

        if st.button("Predict Diabetes"):
            input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                      insulin, bmi, diabetes_pedigree, age]],
                                    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            prediction = diabetes_model.predict(input_df)
            prob = diabetes_model.predict_proba(input_df)

            result = "You **have** diabetes." if prediction[0] == 1 else "You **do not have** diabetes."
            st.markdown(f"### 🧾 Prediction Result: {result}")
            st.info(f"Probability of having diabetes: `{prob[0][1] * 100:.2f}%`")
            st.info(f"Probability of not having diabetes: `{prob[0][0] * 100:.2f}%`")

    elif model_selection == "Heart Disease Prediction":
        st.header("❤️ Heart Disease Prediction Inputs")

        with st.expander("ℹ️ Safe Ranges Reference"):
            st.markdown("""
            - **Blood Pressure**: <120/80 mm Hg
            - **Cholesterol**: <200 mg/dL desirable
            - **Max Heart Rate**: ~220 - age
            - **Oldpeak**: ≤2.0 generally normal
            """)

        age = dual_input("Age", 0, 120, 40, step=1, help_text="Heart disease risk increases with age, especially after 45.")
        sex = st.selectbox("Sex", ['Male', 'Female'])
        chest_pain_type = st.selectbox("Chest Pain Type (e.g., Typical Angina)", ['Typical Angina','Heart Burn', 'Anxiety Attack'])
        resting_bp = dual_input("Resting Blood Pressure (mm Hg)", 0, 200, 120, step=1, help_text="Normal: <120 mm Hg. Elevated: 120–129. High: 130+.")
        cholesterol = dual_input("Cholesterol (mg/dl)", 0, 500, 200, step=1, help_text="Desirable: <200 mg/dL. Borderline high: 200–239. High: 240+")
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ['Yes', 'No'])
        resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
        max_hr = dual_input("Max Heart Rate Achieved (bpm)", 0, 220, 150, step=1, help_text="Estimated max HR = 220 - age. Higher peak is generally better.")
        exercise_angina = st.selectbox("Exercise Angina?", ['Yes', 'No'])
        oldpeak = dual_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1, is_float=True, help_text="ST depression induced by exercise. Higher may indicate ischemia.")
        st_slope = st.selectbox("ST Segment Slope", ['Upsloping', 'Flat', 'Downsloping'])

        input_data = {
            'Age': age,
            'Resting BP': resting_bp,
            'Cholesterol': cholesterol,
            'Fasting BS': fasting_bs,
            'Max HR': max_hr,
            'Oldpeak': oldpeak,
            'Sex': sex,
            'Chest Pain Type': chest_pain_type,
            'Resting ECG': resting_ecg,
            'Exercise Angina': exercise_angina,
            'ST Slope': st_slope,
        }

        if st.button("Visualize Inputs"):
            plot_input_data(input_data, "Heart Disease")

        if st.button("Predict Heart Disease"):
            input_dict = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': 1 if fasting_bs == 'Yes' else 0,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex': sex,
                'ChestPainType': chest_pain_type,
                'RestingECG': resting_ecg,
                'ExerciseAngina': exercise_angina,
                'ST_Slope': st_slope,
            }

            input_df = pd.DataFrame([input_dict])
            input_df = pd.get_dummies(input_df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

            model_features = heart_model.feature_names_in_
            for feature in model_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[model_features]

            prediction = heart_model.predict(input_df)
            prob = heart_model.predict_proba(input_df)

            result = "You **have** heart disease." if prediction[0] == 1 else "You **do not have** heart disease."
            st.markdown(f"### 🧾 Prediction Result: {result}")
            st.info(f"Probability of having heart disease: `{prob[0][1] * 100:.2f}%`")
            st.info(f"Probability of not having heart disease: `{prob[0][0] * 100:.2f}%`")

if __name__ == "__main__":
    main()
