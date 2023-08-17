import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata

import random

import flor

# Device configuration
device = torch.device(
    flor.arg("device", "cuda" if torch.cuda.is_available() else "cpu")
)

seed = flor.arg("seed", default=random.randint(1, 99))
torch.manual_seed(seed)

# Hyper-parameters
input_size = 784
hidden_size = flor.arg("hidden", default=500)
num_classes = 10
num_epochs = flor.arg("epochs", 5)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="../data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="../data", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torchdata.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torchdata.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print_every = flor.arg("print_every", 500)

with flor.checkpointing(model=model, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        for i, (images, labels) in flor.loop("step", enumerate(train_loader)):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_every == 0:
                flor.log("loss", loss.item())

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    flor.log("accuracy", 100 * correct / total)
    flor.log("correct", correct)
