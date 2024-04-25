import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def filter_by_class(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

classes = random.sample(range(10), 2)
filtered_train_dataset = filter_by_class(train_dataset, classes)
filtered_test_dataset = filter_by_class(test_dataset, classes)

train_loader = DataLoader(filtered_train_dataset, batch_size=len(filtered_train_dataset), shuffle=True)
test_loader = DataLoader(filtered_test_dataset, batch_size=len(filtered_test_dataset), shuffle=False)

# Extract features and labels from loaders
def extract_features_and_labels(loader):
    for images, labels in loader:
        features = images.view(images.size(0), -1).numpy()
        labels = labels.numpy()
        return features, labels

x_train, y_train = extract_features_and_labels(train_loader)
x_test, y_test = extract_features_and_labels(test_loader)

# Define a pipeline with feature extraction and SVM classifier
pipeline = Pipeline([
    ('feature_extraction', PCA(n_components=50)),  
    ('clf', SVC(kernel='linear', random_state=42))  
])

param_grid = {
    'feature_extraction': [PCA(n_components=50), LDA(n_components=1)],
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear'] 
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=3)

# Perform the grid search
grid_search.fit(x_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
predictions = best_model.predict(x_test)
validation_accuracy = accuracy_score(y_test, predictions) * 100
print(f"Validation Accuracy: {validation_accuracy:.2f}%")
print(classification_report(y_test, predictions))
