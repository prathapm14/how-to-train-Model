# How to Train Models in Azure Machine Learning

A comprehensive guide to training machine learning models using Azure Machine Learning service.

## 📑 Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Setup Azure ML Workspace](#setup-azure-ml-workspace)
- [Training Methods](#training-methods)
- [Code Examples](#code-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Introduction

Azure Machine Learning is a cloud-based service for creating and managing machine learning solutions. It provides tools and services to:

- **Train models** at scale using powerful compute resources
- **Track experiments** and metrics
- **Deploy models** as web services
- **Manage ML lifecycle** end-to-end
- **Collaborate** with team members

### Key Benefits
✅ Scalable compute infrastructure  
✅ Built-in MLOps capabilities  
✅ Integration with popular ML frameworks  
✅ Automated machine learning (AutoML)  
✅ Responsible AI tools  

---

## Prerequisites

### 1. Azure Account
- Active Azure subscription ([Create free account](https://azure.microsoft.com/free/))
- Azure Machine Learning workspace

### 2. Development Environment
```bash
# Install Azure ML SDK
pip install azureml-sdk

# Install additional packages
pip install azureml-core azureml-dataset-runtime azureml-train-core
```

### 3. Required Tools
- Python 3.7 or higher
- Azure CLI with ML extension
- Visual Studio Code (recommended)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Azure ML Workspace                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │              │    │              │    │              │    │
│  │  Datasets    │───▶│   Training   │───▶│    Models    │    │
│  │              │    │   Compute    │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │              │    │              │    │              │    │
│  │ Datastores   │    │ Experiments  │    │ Endpoints    │    │
│  │              │    │              │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Training Workflow

```
┌─────────────┐
│   Prepare   │
│    Data     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Configure  │
│ Environment │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Create    │
│Training Job │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Submit    │
│  & Monitor  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Register   │
│    Model    │
└─────────────┘
```

---

## Setup Azure ML Workspace

### Method 1: Using Azure Portal

1. Navigate to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource" → Search "Machine Learning"
3. Fill in the details:
   - Subscription
   - Resource group
   - Workspace name
   - Region
4. Click "Review + Create"

### Method 2: Using Azure CLI

```bash
# Login to Azure
az login

# Create resource group
az group create --name ml-rg --location eastus

# Create ML workspace
az ml workspace create --name my-ml-workspace \
    --resource-group ml-rg \
    --location eastus
```

### Method 3: Using Python SDK

```python
from azureml.core import Workspace

ws = Workspace.create(
    name='my-ml-workspace',
    subscription_id='<your-subscription-id>',
    resource_group='ml-rg',
    create_resource_group=True,
    location='eastus'
)

# Save workspace configuration
ws.write_config()
```

---

## Training Methods

Azure ML supports multiple training approaches:

### 1. **Script-based Training** (Recommended)
- Write custom Python training scripts
- Full control over training logic
- Use any ML framework (scikit-learn, PyTorch, TensorFlow)

### 2. **Automated ML (AutoML)**
- Automatically try multiple algorithms
- Hyperparameter tuning
- Best for quick prototyping

### 3. **Designer** (Low-code)
- Drag-and-drop interface
- Visual pipeline creation
- No coding required

---

## Code Examples

### Example 1: Simple Scikit-learn Training

#### Training Script (`train.py`)

```python
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
```

#### Submit Training Job (`submit_training.py`)

```python
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Connect to workspace
ws = Workspace.from_config()
print(f"Connected to workspace: {ws.name}")

# Create or get compute target
compute_name = "cpu-cluster"
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print(f"Found existing compute target: {compute_name}")
except ComputeTargetException:
    print(f"Creating new compute target: {compute_name}")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_D2_V2',
        max_nodes=4,
        idle_seconds_before_scaledown=300
    )
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Create environment
env = Environment.from_conda_specification(
    name='sklearn-env',
    file_path='environment.yml'
)

# Configure training job
config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    arguments=[
        '--n-estimators', 200,
        '--max-depth', 15
    ],
    compute_target=compute_target,
    environment=env
)

# Submit experiment
experiment = Experiment(workspace=ws, name='sklearn-training-experiment')
run = experiment.submit(config)

print(f"Run submitted. Run ID: {run.id}")
print(f"Monitor at: {run.get_portal_url()}")

# Wait for completion
run.wait_for_completion(show_output=True)

# Get metrics
metrics = run.get_metrics()
print("\nTraining Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Register model
model = run.register_model(
    model_name='sklearn-rf-model',
    model_path='outputs/model.pkl',
    description='Random Forest classifier trained on iris dataset'
)
print(f"\nModel registered: {model.name} (version {model.version})")
```

---

### Example 2: PyTorch Deep Learning

#### Training Script (`train_pytorch.py`)

```python
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from azureml.core import Run

# Get run context
run = Run.get_context()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=0.001)
args = parser.parse_args()

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Dummy data (replace with actual data loading)
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 3, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(input_size=20, hidden_size=50, num_classes=3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    # Log metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    run.log('loss', avg_loss)
    run.log('accuracy', accuracy)
    
    print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Save model
os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/pytorch_model.pth')
print("Model saved!")

run.complete()
```

---

### Example 3: Using Azure ML CLI v2

#### Job Configuration (`job.yml`)

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: ./src
command: python train.py --n-estimators ${{inputs.n_estimators}} --max-depth ${{inputs.max_depth}}
environment: azureml:sklearn-env@latest
compute: azureml:cpu-cluster
experiment_name: sklearn-cli-experiment
description: Training Random Forest model using CLI v2

inputs:
  n_estimators:
    type: integer
    default: 100
  max_depth:
    type: integer
    default: 10

outputs:
  model_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/models/
```

#### Submit Job

```bash
# Submit the job
az ml job create -f job.yml

# Stream logs
az ml job stream --name <job-name>

# Download outputs
az ml job download --name <job-name> --output-name model_output
```

---

### Example 4: Automated ML (AutoML)

```python
from azureml.core import Workspace, Dataset, Experiment
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import ComputeTarget

# Connect to workspace
ws = Workspace.from_config()

# Get or create dataset
dataset = Dataset.get_by_name(ws, name='training-data')

# Configure AutoML
automl_config = AutoMLConfig(
    task='classification',
    primary_metric='accuracy',
    training_data=dataset,
    label_column_name='label',
    compute_target='cpu-cluster',
    n_cross_validations=5,
    iterations=20,
    max_concurrent_iterations=4,
    experiment_timeout_hours=1,
    enable_early_stopping=True,
    featurization='auto',
    enable_onnx_compatible_models=True
)

# Submit experiment
experiment = Experiment(ws, 'automl-classification')
run = experiment.submit(automl_config, show_output=True)

# Get best model
best_run, fitted_model = run.get_output()
print(f"Best Run ID: {best_run.id}")
print(f"Best Model: {fitted_model}")

# Register best model
model = best_run.register_model(
    model_name='automl-best-model',
    description='Best model from AutoML run'
)
``` 

---

## Best Practices

### 1. **Compute Management**
```python
# Use auto-scale compute clusters
compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_D2_V2',
    min_nodes=0,  # Scale to 0 when idle
    max_nodes=4,
    idle_seconds_before_scaledown=300  # 5 minutes
)
```

### 2. **Experiment Tracking**
```python
# Log all important metrics
run.log('metric_name', value)
run.log_list('metric_list', [1, 2, 3])
run.log_row('metric_row', col1=1, col2=2)
run.log_table('metric_table', {'col1': [1, 2], 'col2': [3, 4]})

# Log artifacts
run.upload_file('outputs/chart.png', 'path/to/chart.png')
```

### 3. **Data Management**
```python
# Register datasets for versioning
from azureml.core import Dataset

dataset = Dataset.Tabular.from_delimited_files(path='data/*.csv')
dataset = dataset.register(
    workspace=ws,
    name='training-data',
    description='Training dataset v1',
    create_new_version=True
)
```

### 4. **Environment Management**
```python
# Use curated environments when possible
from azureml.core import Environment

env = Environment.get(workspace=ws, name='AzureML-sklearn-0.24-ubuntu18.04-py37-cpu')

# Or create custom environment
env = Environment.from_dockerfile(name='custom-env', dockerfile='./Dockerfile')
```

### 5. **Cost Optimization**
- Use low-priority VMs for non-urgent training
- Enable auto-shutdown for compute instances
- Use dataset caching
- Clean up unused resources

```python
# Use low-priority VMs
compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_D2_V2',
    vm_priority='lowpriority',  # Cheaper but can be preempted
    max_nodes=4
)
```

---

## Monitoring and Debugging

### View Run Details

```python
from azureml.core import Experiment, Run

# Get experiment
experiment = Experiment(workspace=ws, name='my-experiment')

# List all runs
for run in experiment.get_runs():
    print(f"Run ID: {run.id}, Status: {run.status}")
    print(f"Metrics: {run.get_metrics()}")

# Get specific run
run = Run(experiment, run_id='<run-id>')
print(run.get_details())
print(run.get_metrics())
print(run.get_file_names())
```

### Download Logs

```python
# Download all logs
run.download_files(prefix='logs/', output_directory='./logs')

# View specific log
run.get_file_names()
```

### Real-time Monitoring

```python
# Stream logs in real-time
run.wait_for_completion(show_output=True)

# Or use the portal URL
print(f"Monitor at: {run.get_portal_url()}")
```

---

## Troubleshooting

### Common Issues

#### 1. **Compute Target Not Found**
```python
# Always check if compute exists before using
try:
    compute_target = ComputeTarget(workspace=ws, name='cpu-cluster')
except ComputeTargetException:
    # Create it if it doesn't exist
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', max_nodes=4)
    compute_target = ComputeTarget.create(ws, 'cpu-cluster', compute_config)
    compute_target.wait_for_completion(show_output=True)
```

#### 2. **Environment Build Failures**
```python
# Test environment locally first
env = Environment.from_conda_specification(name='test-env', file_path='environment.yml')
env.build_local(workspace=ws, useDocker=True)
```

#### 3. **Authentication Issues**
```bash
# Use interactive login
az login

# Or use service principal
az login --service-principal \
    --username <app-id> \
    --password <password> \
    --tenant <tenant-id>
```

#### 4. **Out of Memory Errors**
- Use larger VM sizes
- Reduce batch size
- Enable data streaming instead of loading all in memory

#### 5. **Slow Data Loading**
- Use Dataset caching
- Use faster storage (Premium SSD)
- Parallelize data loading

---

## Additional Resources

### Official Documentation
- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Python SDK Reference](https://docs.microsoft.com/python/api/overview/azure/ml/)
- [CLI v2 Reference](https://docs.microsoft.com/cli/azure/ml)

### Sample Repositories
- [Azure ML Examples](https://github.com/Azure/azureml-examples)
- [Azure ML Notebooks](https://github.com/Azure/MachineLearningNotebooks)

### Learning Paths
- [Microsoft Learn - Azure ML](https://docs.microsoft.com/learn/paths/build-ai-solutions-with-azure-ml-service/)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

## Contact

For questions or feedback, please open an issue in this repository.

**Happy Training! 🚀**