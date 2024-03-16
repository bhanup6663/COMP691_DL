# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14MQbszr8IC26QaWt_D1XuuCcQTmRkPhY
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Normalization parameters from the ImageNet dataset that pretrained models use
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Data preprocessing with augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Data preprocessing for validation/testing
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Load CIFAR-10 dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Define your two classes indices based on the CIFAR-10 dataset
class_indices = [0, 1]  # Replace with your chosen classes

# Now, filter out the data for the two selected classes
train_indices = [i for i, label in enumerate(train_data.targets) if label in class_indices]
test_indices = [i for i, label in enumerate(test_data.targets)]

# Split training indices for training and validation sets
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42, stratify=[train_data.targets[i] for i in train_indices])

# Create subsets for train, validation, and test
train_subset = Subset(train_data, train_indices)
val_subset = Subset(train_data, val_indices)
test_subset = Subset(test_data, test_indices)

# DataLoaders for the datasets
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

from torchvision import models

# Load a pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Replace the last fully connected layer with a binary classifier
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Transfer the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights

        scheduler.step()  # Update learning rate

        print(f'Epoch {epoch+1}/{num_epochs} completed.')
    return model

# Train the model
model_ft = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)
    accuracy = corrects / total
    print(f'Test Accuracy: {accuracy:.4f}')

# Evaluate the model
evaluate_model(model_ft, test_loader)

