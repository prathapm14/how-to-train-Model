import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from azureml.core import Run

# Get run context
run = Run.get_context()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
parser.add_argument('--max-depth', type=int, default=10, help='Max depth of trees')
parser.add_argument('--data-path', type=str, help='Path to training data')
args = parser.parse_args()

print(f"Training with n_estimators={args.n_estimators}, max_depth={args.max_depth}")

# Load data (example with dummy data)
# Replace this with your actual data loading logic
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Log metrics
run.log('accuracy', accuracy)
run.log('precision', precision)
run.log('recall', recall)
run.log('n_estimators', args.n_estimators)
run.log('max_depth', args.max_depth)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Save model
os.makedirs('outputs', exist_ok=True)
model_path = 'outputs/model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

run.complete()