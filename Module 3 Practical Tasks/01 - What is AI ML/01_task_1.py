# Task 1: Explore the Iris Dataset End-to-End

# Load the Iris dataset using sklearn.datasets.load_iris(). Split the data into 70% training and 30% testing. 
# Train three different classifiers: Logistic Regression, K-Nearest Neighbors (k=5), and a Support Vector Machine. 
# Compare their accuracies and write a short analysis (3-5 sentences) explaining which model performs best and why you think that is.


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine (RBF)": SVC(kernel='rbf', random_state=42)
}

print("Model Accuracy Comparison:")
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:30s}: {acc:.2%}")