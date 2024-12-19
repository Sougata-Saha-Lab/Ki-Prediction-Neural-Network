import os
import csv
from Bio.PDB import PDBParser

# Define the folder containing PDB files and output CSV file
pdb_folder = "Peptide"
output_csv = "Peptide.csv"

# Create a PDB parser object
parser = PDBParser(QUIET=True)

# Function to extract details from a PDB file and save them into a list
def extract_pdb_details(file_path):
    details = []
    structure = parser.get_structure(os.path.basename(file_path), file_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                res_id = residue.id[1]
                for atom in residue:
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    details.append({
                        "File": os.path.basename(file_path),
                        "Model": model.id,
                        "Chain": chain.id,
                        "Residue": res_name,
                        "Residue_ID": res_id,
                        "Atom": atom_name,
                        "X": coord[0],
                        "Y": coord[1],
                        "Z": coord[2]
                    })
    return details

# List to hold all extracted details
all_details = []

# Iterate over all PDB files in the folder
for file_name in os.listdir(pdb_folder):
    if file_name.endswith(".pdb"):
        file_path = os.path.join(pdb_folder, file_name)
        all_details.extend(extract_pdb_details(file_path))

# Save the details to a CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    fieldnames = ["File", "Model", "Chain", "Residue", "Residue_ID", "Atom", "X", "Y", "Z"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_details)

print(f"Details of all PDB files have been saved to {output_csv}.")

