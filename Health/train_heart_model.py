import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the heart disease dataset
data = pd.read_csv('heart-dataset.csv')
data.columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']

# Convert categorical columns into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'RestingECG', 
                                     'ExerciseAngina', 'ST_Slope'], drop_first=True)

# Features and target variable
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Save the model
with open('heart_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Heart Disease model trained and saved as heart_model.pkl")
