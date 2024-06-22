import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Parameters for synthetic data generation
n_samples = 1000  # Number of samples
n_features = 10   # Number of features
n_informative = 5  # Number of informative features
n_redundant = 2   # Number of redundant features
random_state = 42  # Random seed for reproducibility

# Generate synthetic data
X, y = make_classification(n_samples=n_samples, 
                           n_features=n_features, 
                           n_informative=n_informative, 
                           n_redundant=n_redundant, 
                           random_state=random_state)

# Convert to pandas DataFrame
feature_columns = [f'feature_{i}' for i in range(1, n_features + 1)]
df = pd.DataFrame(X, columns=feature_columns)
df['default'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
log_reg = LogisticRegression(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

# Train models
log_reg.fit(X_train_scaled, y_train)
random_forest.fit(X_train, y_train)  # No scaling needed for tree-based methods
gradient_boosting.fit(X_train, y_train)  # No scaling needed for tree-based methods

# Make predictions
y_pred_proba_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_random_forest = random_forest.predict_proba(X_test)[:, 1]
y_pred_proba_gradient_boosting = gradient_boosting.predict_proba(X_test)[:, 1]

# Evaluate models
def evaluate_model(y_test, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
    precision = precision_score(y_test, y_pred_proba > 0.5)
    recall = recall_score(y_test, y_pred_proba > 0.5)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, precision, recall, roc_auc

log_reg_results = evaluate_model(y_test, y_pred_proba_log_reg)
random_forest_results = evaluate_model(y_test, y_pred_proba_random_forest)
gradient_boosting_results = evaluate_model(y_test, y_pred_proba_gradient_boosting)

print("Logistic Regression Results:", log_reg_results)
print("Random Forest Results:", random_forest_results)
print("Gradient Boosting Results:", gradient_boosting_results)

# Choose the best model based on evaluation (for simplicity, let's assume it's Gradient Boosting)
best_model = gradient_boosting
best_model_proba = y_pred_proba_gradient_boosting

# Predict credit risk score for each applicant in the test set
X_test_with_proba = pd.DataFrame(X_test, columns=feature_columns)
X_test_with_proba['credit_risk_score'] = best_model_proba

# Save the results to a CSV file
output_csv_file = 'credit_risk_scores.csv'
X_test_with_proba.to_csv(output_csv_file, index=False)

# Display the table
print(X_test_with_proba.head(10))  # Display the first 10 rows as a sample
