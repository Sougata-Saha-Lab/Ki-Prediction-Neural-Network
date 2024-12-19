import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcTPSA, CalcNumHeteroatoms

# Load the input CSV file
input_file = "compounds_smiles.csv"
output_file = "compounds_smiles_with_descriptors.csv"

# Read the CSV file
data = pd.read_csv(input_file)

# Check for a column with SMILES strings
if "SMILES" not in data.columns:
    raise ValueError("The input file must contain a 'SMILES' column.")

# Function to calculate molecular features from SMILES
def extract_smiles_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        return {
            "Molecular_Formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "Molecular_Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": CalcTPSA(mol),
            "Num_H_Bond_Donors": CalcNumHBD(mol),
            "Num_H_Bond_Acceptors": CalcNumHBA(mol),
            "Num_Rotatable_Bonds": Descriptors.NumRotatableBonds(mol),
            "Num_Atoms": mol.GetNumAtoms(),
            "Num_Heteroatoms": CalcNumHeteroatoms(mol),
            "Formal_Charge": Chem.GetFormalCharge(mol),
            "Chirality_Centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "Num_Rings": Descriptors.RingCount(mol)
        }
    except Exception as e:
        # Return NaN for invalid SMILES
        return {
            "Molecular_Formula": None,
            "Molecular_Weight": None,
            "LogP": None,
            "TPSA": None,
            "Num_H_Bond_Donors": None,
            "Num_H_Bond_Acceptors": None,
            "Num_Rotatable_Bonds": None,
            "Num_Atoms": None,
            "Num_Heteroatoms": None,
            "Formal_Charge": None,
            "Chirality_Centers": None,
            "Num_Rings": None
        }

# Apply the function to the SMILES column
features = data["SMILES"].apply(extract_smiles_features)

# Convert the extracted features to a DataFrame
features_df = pd.DataFrame(features.tolist())

# Merge the original data with the new features
merged_data = pd.concat([data, features_df], axis=1)

# Save the updated data to a new CSV file
merged_data.to_csv(output_file, index=False)

print(f"Details extracted and saved to {output_file}.")

