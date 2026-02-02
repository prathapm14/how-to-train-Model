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
y_train = torch.randint(0, 3, (1000,))	rain_dataset = TensorDataset(X_train, y_train)	rain_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

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