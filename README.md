# Ki-Prediction-Neural-Network
Prediction of Ki values using Neural Network 

## Overview
This repository contains four Python scripts designed for molecular descriptor extraction, clustering, neural network-based Ki prediction, and PDB file analysis. Below is a detailed description of each script and their usage.

---

## 1. **SMILES Descriptor Extraction**

### File: `main_script1.py`

### Purpose:
This script extracts molecular descriptors from SMILES strings and appends them to the input dataset.

### Input:
- CSV file with a `SMILES` column (e.g., `compounds_smiles.csv`).

### Output:
- A CSV file (`compounds_smiles_with_descriptors.csv`) containing molecular descriptors such as molecular weight, LogP, TPSA, and others.

### How to Run:
```bash
python main_script1.py
```
Ensure the input file name is correctly set in the script (`input_file` variable).

---

## 2. **PDB File Analysis**

### File: `main_script2.py`

### Purpose:
Extracts atom-level details from PDB files, including residue information and atomic coordinates.

### Input:
- Folder containing `.pdb` files (e.g., `Peptide`).

### Output:
- A CSV file (`Peptide.csv`) with details such as residue names, IDs, atom names, and coordinates.

### How to Run:
```bash
python main_script2.py
```
Ensure the `pdb_folder` variable is set to the directory containing your `.pdb` files.

---

## 3. **Clustering and Neural Network Modeling**

### File: `Model.py`

### Purpose:
Performs clustering on compound descriptors using KMeans and builds a neural network model to predict Ki values with hyperparameter tuning.

### Input:
- `reference_compounds_with_descriptors.csv`: A CSV file containing molecular descriptors and Ki values.
- `PDB_inhibitor.csv`: Optional file for additional analysis.

### Output:
- Visualizations of clusters and learning curves.
- A saved neural network model (`neural_network_model.pkl`) and scaler (`scaler.pkl`).

### How to Run:
```bash
python Model.py
```
Ensure the input file names are set correctly in the script.

---

## 4. **Predicting Ki for New Compounds**

### File: `Result.py`

### Purpose:
Uses the saved neural network model to predict Ki values for new compounds based on their molecular descriptors.

### Input:
- `compounds_smiles_with_descriptors.csv`: A CSV file containing molecular descriptors for new compounds.
- `reference_compounds_with_descriptors.csv`: Used to determine reasonable Ki value ranges.

### Output:
- A CSV file (`Noble_Ki_values.csv`) containing predicted Ki values.
- Visualizations of predictions.

### How to Run:
```bash
python Result.py
```
Ensure the input file names match those in the script.

---

## Requirements
- Python 3.7+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - rdkit
  - Bio.PDB
  - joblib

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Workflow
1. **SMILES Descriptor Extraction**:
   Run `main_script1.py` to extract molecular descriptors from SMILES strings.

2. **PDB File Analysis**:
   Run `main_script2.py` to analyze PDB files and extract atomic details.

3. **Model Training and Clustering**:
   Use `Model.py` to cluster compounds and train a neural network model.

4. **Ki Prediction**:
   Use `Result.py` to predict Ki values for new compounds.

---

## Notes
- Ensure file paths and input file names are correctly set in each script.
- For large datasets, consider optimizing the scripts for performance.

##  Authors & Contributors  

- [Rahul Naga](https://github.com/RAHUL-NAGA)  
- [Dr. Sougata Saha](https://github.com/Sougata-Saha-Lab)  

## License
This project is licensed under the MIT License. See the LICENSE file for details.


