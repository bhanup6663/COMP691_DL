import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

# Extract and preprocess data
def extract_data(loader):
    for data in loader:
        images, labels = data
        # Flatten images and move data to numpy
        images = images.numpy().reshape(images.size(0), -1)
        labels = labels.numpy()
        return images, labels

x_train, y_train = extract_data(trainloader)
x_test, y_test = extract_data(testloader)

def select_random_classes(y_train, num_classes=2):
    classes = np.unique(y_train)
    return random.sample(list(classes), num_classes)

train_classes = select_random_classes(y_train)
train_indices = np.where((y_train == train_classes[0]) | (y_train == train_classes[1]))[0]
x_train, y_train = x_train[train_indices], y_train[train_indices]

x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=50, stratify=y_train)
y_train = y_train.ravel()

x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=10, stratify=y_test)
y_test = y_test.ravel()

# Simulate training "epochs"
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)
    
# Predict and evaluate
predictions = svm_model.predict(x_test)
print(classification_report(y_test, predictions))
