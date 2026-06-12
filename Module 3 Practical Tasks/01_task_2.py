# Task 2: Visualize Overfitting with Polynomial Regression

# Generate a synthetic dataset using np.sin() with added Gaussian noise (50 samples). Fit polynomial regression models of degree 1, 3, 5, 10, and 20. 
# For each degree, compute and record both the training MSE and test MSE (using a held-out test set). 
# Create a table or text output showing how training error decreases with complexity while test error eventually increases. 
# Identify which degree represents the best bias-variance tradeoff.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

synthetic_set = np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.5, 50)
X = np.arange(50).reshape(-1, 1)
y = synthetic_set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

degrees = [1, 3, 5, 10, 20]
results = []

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    results.append((degree, train_mse, test_mse))

print("Degree | Training MSE | Test MSE")
for degree, train_mse, test_mse in results:
    print(f"{degree:6d} | {train_mse:.4f}       | {test_mse:.4f}")