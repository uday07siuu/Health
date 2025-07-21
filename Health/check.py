import pickle
with open('breast_cancer_rf_model.pkl', 'rb') as file:
    breast_cancer_rf_model = pickle.load(file)
print(breast_cancer_rf_model.n_features_in_)
