# -*- coding: utf-8 -*-
"""
Created on Tue May  6 01:21:37 2025

@author: Mohamed
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv('cleaned_file.csv')
# Function to normalize hand landmarks while keeping labels unchanged
def normalize_hand_landmarks(row):
    # Step 1: Wrist coordinates (New Origin)
    x_wrist, y_wrist = row["x1"], row["y1"]
    
    # Step 2: Middle fingertip coordinates (Scaling reference)
    x_tip, y_tip = row["x9"], row["y9"]

    # Compute scaling factor (distance from wrist to middle fingertip)
    scale = np.sqrt((x_tip - x_wrist) ** 2 + (y_tip - y_wrist) ** 2)
    
    # Avoid division by zero
    if scale == 0:
        return row  # Return original if scaling factor is zero
    
    # Create a copy to prevent modifying original row
    normalized_row = row.copy()
    
    # Normalize all x, y coordinates (excluding labels)
    for i in range(1, 22):  # Loop through 21 landmarks
        normalized_row[f"x{i}"] = (row[f"x{i}"] - x_wrist) / scale
        normalized_row[f"y{i}"] = (row[f"y{i}"] - y_wrist) / scale
    
    return normalized_row

# Apply normalization to all rows
df.iloc[:, :-1] = df.iloc[:, :-1].apply(normalize_hand_landmarks, axis=1)

# Save the processed dataset
df.to_csv("normalized_ٍِِِِASL.csv", index=False)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['letter'] = label_encoder.fit_transform(df['letter'])
df['letter'].unique()


# 2. Split features and labels
X = df.drop('letter', axis=1)
Y = df['letter']

# 3. Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# 4. Initialize RandomForest
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],          # Number of trees
    'max_depth': [10, 20, 30, None],          # Tree depth
    'min_samples_split': [2, 5, 10],          # Min samples to split an internal node
    'min_samples_leaf': [1, 2, 4],            # Min samples at a leaf node
    'max_features': ['sqrt', 'log2', None],   # Features considered at each split
    'bootstrap': [True, False]                # Whether to bootstrap samples
}
random_search = RandomizedSearchCV(rf, param_distributions = param_grid,\
                                  n_iter = 5, cv=3, verbose = 2, n_jobs = -1 \
                                  , random_state = 42)
random_search.fit(X_train, Y_train)

print("Best Hyperparameters:", random_search.best_params_)
# Best Hyperparameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 20, 'bootstrap': False}
best_rf = random_search.best_estimator_

# Train Accuracy 
y_pred_train = best_rf.predict(X_train)
train_accuracy = accuracy_score(Y_train, y_pred_train)
print(f"Optimized Train Accuracy: {train_accuracy * 100:.2f}%")
# Optimized Train Accuracy: 99.98%
# Test accuracy
y_pred_test = best_rf.predict(X_test)
test_accuracy = accuracy_score(Y_test, y_pred_test)
print(f"Optimized Test Accuracy: {test_accuracy * 100:.2f}%") 
#Optimized Test Accuracy: 98.21%

import joblib

# Save the trained model
joblib.dump(best_rf, 'asl_random_forest_model.joblib')

# Save the label encoder as well (to convert predicted label index back to letter)
joblib.dump(label_encoder, 'asl_label_encoder.joblib')