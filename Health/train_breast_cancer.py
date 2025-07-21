import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
data = pd.read_csv('breast-cancer.csv.csv')

# Preprocess the dataset
# Encode the target column ('diagnosis') as binary (M=1, B=0)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Separate features (X) and target (y)
X = data.drop(columns=['id', 'diagnosis'])  # Drop 'id' and target column
y = data['diagnosis']

# Handle imbalanced classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the SVM model
model = SVC(probability=True, kernel='linear', random_state=42)  # You can change kernel if needed
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Calculate ROC-AUC score
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
roc_auc = roc_auc_score(y_test, y_prob)
print(f'ROC-AUC Score: {roc_auc:.2f}')

# Save the trained model to a file named 'breast_cancer_svm_model.pkl'
with open('breast_cancer_svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Breast cancer SVM model trained and saved as breast_cancer_svm_model.pkl")
