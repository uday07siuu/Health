import streamlit as st
import numpy as np
import pickle
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the trained models
with open('diabetes_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('heart_model.pkl', 'rb') as file:
    heart_model = pickle.load(file)


def main():
    st.title("Health Prediction App")
    st.sidebar.title("Select Prediction Model")

    # Sidebar for model selection
    model_selection = st.sidebar.selectbox(
        "Choose a model", ("Diabetes Prediction", "Heart Disease Prediction")
    )

    if model_selection == "Diabetes Prediction":
        # Input fields for diabetes model
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose Level (mg/dl)", min_value=0, max_value=300)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100)
        insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=800)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
        age = st.number_input("Age", min_value=0, max_value=120)

        if st.button("Predict Diabetes"):
            diabetes_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            diabetes_prediction = diabetes_model.predict(diabetes_input)
            diabetes_probabilities = diabetes_model.predict_proba(diabetes_input)
            st.success(f"The model predicts: You {'have' if diabetes_prediction[0] == 1 else 'do not have'} diabetes.")
            st.info(f"Probability of having diabetes: {diabetes_probabilities[0][1] * 100:.2f}%")
            st.info(f"Probability of not having diabetes: {diabetes_probabilities[0][0] * 100:.2f}%")

    elif model_selection == 'Heart Disease Prediction':
        # Collect user inputs for heart disease prediction
        age = st.number_input('Age', min_value=0)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        chest_pain_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 
                                                             'Non-Anginal Pain', 'Asymptomatic'])
        resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0)
        cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0)
        fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['Yes', 'No'])
        resting_ecg = st.selectbox('Resting Electrocardiographic Results', 
                                    ['Normal', 'ST-T wave abnormality', 
                                     'Left ventricular hypertrophy'])
        max_hr = st.number_input('Maximum Heart Rate Achieved (bpm)', min_value=0)
        exercise_angina = st.selectbox('Exercise Angina?', ['Yes', 'No'])
        oldpeak = st.number_input('Oldpeak (depression induced by exercise relative to rest)', min_value=0.0)
        st_slope = st.selectbox('Slope of ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

        if st.button('Predict Heart Disease'):
            # Create a DataFrame with input values
            input_data = {
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

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # One-hot encode the categorical columns (matching the training process)
            input_df = pd.get_dummies(input_df, columns=['Sex',
                                                          'ChestPainType',
                                                          'RestingECG',
                                                          'ExerciseAngina',
                                                          'ST_Slope'])

            # Align with training features
            model_features = heart_model.feature_names_in_
            for feature in model_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Add missing feature with default value 0

            # Ensure input_df has the same column order as the training data
            input_df = input_df[model_features]

            # Make predictions
            prediction = heart_model.predict(input_df)
            probabilities = heart_model.predict_proba(input_df)

            # Display the result
            st.success(f"The model predicts: You {'have' if prediction[0] == 1 else 'do not have'} heart disease.")
            st.info(f"Probability of having heart disease: {probabilities[0][1] * 100:.2f}%")
            st.info(f"Probability of not having heart disease: {probabilities[0][0] * 100:.2f}%")

if __name__ == "__main__":
    main()
