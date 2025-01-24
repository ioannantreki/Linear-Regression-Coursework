# -*- coding: utf-8 -*-
"""linear_regression.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Linear Algebra

# Setting the seed to 2 (last digit of student number)
np.random.seed(2)

# Generate two random integer matrices X and Y of dimensions 3x4 and 4x3 respectively
X = np.random.randint(low=0, high=10, size=(3, 4))
Y = np.random.randint(low=0, high=10, size=(4, 3))

# Generate two random integer vectors g and z of size 4
g = np.random.randint(low=0, high=10, size=4)
z = np.random.randint(low=0, high=10, size=4)

# Vectors g and z
g = np.array(g)
z = np.array(z)

# 1. Inner product of g and z
inner_product = np.dot(g, z)

# 2. Matrix-vector product of X and g
mv_prod1 = np.dot(X, g)

# 3. Matrix/Dot product of X and Y
mv_prod2 = np.dot(X, Y)

# 4. L2/Frobenius norms of the vectors and matrices
norm_X = np.linalg.norm(X, ord='fro')
norm_Y = np.linalg.norm(Y, ord='fro')
norm_g = np.linalg.norm(g, ord=2)
norm_z = np.linalg.norm(z, ord=2)

print("Matrix X (3x4):\n", X)
print("\nMatrix Y (4x3):\n", Y)
print("\nVector g:", g)
print("\nVector z:", z)
print("-" * 80)
print("Inner product of g and z:", inner_product)
print("\nMatrix-vector product X and g:\n", mv_prod1)
print("\nMatrix-vector product X and Y:\n", mv_prod2)
print("-" * 80)
print("Frobenius norm of matrix X:", norm_X)
print("\nFrobenius norm of matrix Y:", norm_Y)
print("\nL2 norm of vector g:", norm_g)
print("\nL2 norm of vector z:", norm_z)

# Train the model

# Importing data
filename = "/content/drive/MyDrive/Colab Notebooks/HousingData.csv"

# Reading data from a CSV file, first row as column headers
data = pd.read_csv(filename)
data.fillna(data.mean(), inplace=True)  # Handle NaNs right after loading

# Data overview/inspect
print(data.head())
print()
print('-' * 100)
print()

# Prepare the data
X = data.drop('MEDV', axis=1)  # Features
y = data['MEDV']               # Target variable

# Define the split ratios
splits = [0.1, 0.2, 0.3, 0.5]  # Including the initial 90/10 split we chose

# Iterate through each split ratio
for split in splits:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

    # Add a column of ones to X_train and X_test to account for the intercept
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Train the model using the normal equations
    theta_best = np.linalg.inv(X_train_b.T @ X_train_b) @ (X_train_b.T @ y_train)

    # Predict the values for training and testing sets
    y_train_pred = X_train_b @ theta_best
    y_test_pred = X_test_b @ theta_best

    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    train_mse = np.mean(train_residuals**2)
    test_mse = np.mean(test_residuals**2)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    # Calculate R-squared
    train_r2 = 1 - np.sum(train_residuals**2) / np.sum((y_train - np.mean(y_train))**2)
    test_r2 = 1 - np.sum(test_residuals**2) / np.sum((y_test - np.mean(y_test))**2)

    # Display the results
    print(f"Train/Test Split: {int((1-split)*100)}/{int(split*100)}")
    print(f"Training RMSE: {train_rmse:.2f}, Training R²: {train_r2:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}, Testing R²: {test_r2:.2f}\n")
