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
