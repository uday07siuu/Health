import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Load the data into a DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Separate features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Evaluate the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Streamlit UI
st.title('Breast Cancer Prediction System')

# Display accuracy
st.subheader('Model Accuracy')
st.write(f"Accuracy on training data: {training_data_accuracy * 100:.2f}%")
st.write(f"Accuracy on test data: {test_data_accuracy * 100:.2f}%")

# Input form for prediction
st.subheader('Enter the features to predict')

# Create input fields for all 30 features
input_data = []
for feature in breast_cancer_dataset.feature_names:
    value = st.number_input(f"{feature}", step=0.01)
    input_data.append(value)

# Prediction button
if st.button('Predict'):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for a single datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_reshaped)
    prediction_probability = model.predict_proba(input_data_reshaped)

    # Display the result
    if prediction[0] == 0:
        st.success('The Breast Cancer is Malignant')
    else:
        st.success('The Breast Cancer is Benign')

    # Display the probabilities
    st.subheader('Prediction Probabilities')
    st.write(f"Probability of being Malignant: {prediction_probability[0][0] * 100:.2f}%")
    st.write(f"Probability of being Benign: {prediction_probability[0][1] * 100:.2f}%")
