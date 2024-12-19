import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load("neural_network_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the new input datasets
new_compounds_df = pd.read_csv("compounds_smiles_with_descriptors.csv")  # Replace with your input file name
reference_compounds_df = pd.read_csv("reference_compounds_with_descriptors.csv")  # Load reference data for IC50

# Define features (same as reference compounds)
features = ['Molecular_Weight', 'LogP', 'TPSA', 'Num_H_Bond_Donors', 'Num_H_Bond_Acceptors',
            'Num_Rotatable_Bonds', 'Num_Atoms', 'Num_Heteroatoms', 'Formal_Charge',
            'Chirality_Centers', 'Num_Rings']

# Prepare features for new compounds (same as reference compounds)
new_compounds_df = new_compounds_df[features].dropna()  # Drop rows with missing values

# Scale the new compounds data using the previously fitted scaler
new_compounds_scaled = scaler.transform(new_compounds_df)

# Predict IC50 values (log-transformed)
predicted_log_ic50 = model.predict(new_compounds_scaled)

# Reverse the log transformation to get the actual IC50 values
predicted_ic50 = np.expm1(predicted_log_ic50)

# Check for reasonable IC50 values
# Get min and max IC50 from the training set (reference_compounds_df)
target = 'IC50Value'  # Ensure this matches the column name for IC50 in your dataset
min_ic50_train = np.min(np.expm1(np.log1p(reference_compounds_df[target])))  # IC50 in training set
max_ic50_train = np.max(np.expm1(np.log1p(reference_compounds_df[target])))  # IC50 in training set

# Apply a reasonable range to the predicted IC50 values
predicted_ic50 = np.clip(predicted_ic50, min_ic50_train, max_ic50_train)  # Clip predictions to the observed range

# Add the predicted IC50 values to the new dataframe
new_compounds_df['Predicted_IC50'] = predicted_ic50

# Display the results (with descriptors and predicted IC50)
print(new_compounds_df[['Molecular_Weight', 'LogP', 'Predicted_IC50']])

# Optionally save the predictions to a CSV, including original compound data
new_compounds_df.to_csv("Noble_ic50_values_with_descriptors.csv", index=False)

# Visualize the predictions
sns.scatterplot(x='Molecular_Weight', y='LogP', hue='Predicted_IC50', data=new_compounds_df, palette='viridis')
plt.title("Predicted IC50 for New Compounds")
plt.show()

