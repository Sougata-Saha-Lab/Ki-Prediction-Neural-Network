import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
pdb_inhibitor_df = pd.read_csv("PDB_inhibitor.csv")
reference_compounds_df = pd.read_csv("reference_compounds_with_descriptors.csv")

# 1. Data Preparation: Merge datasets based on compound name (if needed)
reference_compounds_df = reference_compounds_df.dropna()

# Define features and target
features = ['Molecular_Weight', 'LogP', 'TPSA', 'Num_H_Bond_Donors', 'Num_H_Bond_Acceptors',
            'Num_Rotatable_Bonds', 'Num_Atoms', 'Num_Heteroatoms', 'Formal_Charge',
            'Chirality_Centers', 'Num_Rings']
target = 'IC50Value'

# Prepare features and target
X = reference_compounds_df[features]
y = np.log1p(reference_compounds_df[target])  # Log-transform target to handle skewness

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Clustering using KMeans
n_clusters = 5  # Set number of clusters (can be tuned)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
reference_compounds_df['Cluster'] = clusters

# Visualize Clusters
sns.scatterplot(x='Molecular_Weight', y='LogP', hue='Cluster', data=reference_compounds_df, palette='viridis')
plt.title("KMeans Clustering of Compounds")
plt.show()

# 3. Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Build Neural Network Model with Hyperparameter Tuning
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(max_iter=1000, random_state=42))  # Increased max_iter for convergence
])

# Hyperparameter grid
param_grid = {
    'mlp__hidden_layer_sizes': [(64, 32), (128, 64), (64, 64)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01]  # Trying different learning rates
}

# GridSearchCV for hyperparameter tuning
search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
search.fit(X_train, y_train)

# Best Model
best_model = search.best_estimator_
print("Best Parameters:", search.best_params_)

# 5. Evaluate the Model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print("Train R2 Score:", r2_score(y_train, y_pred_train))
print("Test R2 Score:", r2_score(y_test, y_pred_test))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

# 6. Plotting the Learning Curves
train_errors, test_errors = [], []
for m in range(1, len(X_train)):
    best_model.fit(X_train[:m], y_train[:m])
    y_train_predict = best_model.predict(X_train[:m])
    y_test_predict = best_model.predict(X_test)
    train_errors.append(np.sqrt(mean_squared_error(y_train[:m], y_train_predict)))
    test_errors.append(np.sqrt(mean_squared_error(y_test, y_test_predict)))

plt.plot(np.sqrt(np.array(train_errors)), label="Train RMSE")
plt.plot(np.sqrt(np.array(test_errors)), label="Test RMSE")
plt.legend()
plt.title("Learning Curves")
plt.xlabel("Number of Training Instances")
plt.ylabel("RMSE")
plt.show()

# 7. Cross-validation results
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print("Cross-validation R² scores:", cv_scores)
print("Mean R² score from Cross-validation:", np.mean(cv_scores))

# 8. Check for Overfitting
plt.figure()
sns.kdeplot(y_train, label="Train Actual")
sns.kdeplot(y_pred_train, label="Train Predicted")
sns.kdeplot(y_test, label="Test Actual")
sns.kdeplot(y_pred_test, label="Test Predicted")
plt.legend()
plt.title("Overfitting Check: Actual vs Predicted")
plt.show()

# 9. Predicted vs Actual Scatter Plot
plt.figure()
plt.scatter(y_test, y_pred_test, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual IC50 Values (log-transformed)')
plt.ylabel('Predicted IC50 Values (log-transformed)')
plt.title('Predicted vs Actual IC50 Values')
plt.show()

# 10. Save the Model and Scaler
joblib.dump(best_model, "neural_network_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and Scaler saved successfully.")

