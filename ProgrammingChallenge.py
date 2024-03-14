import csv
import sklearn
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


#read csv
def read_csv_file(file_path):
    data = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


# Load training data from the first CSV file
train_file_path = 'TrainOnMe_orig.csv'
train_data = read_csv_file(train_file_path)

# Load testing data from the second CSV file
test_file_path = 'EvaluateOnMe.csv'
test_data = read_csv_file(test_file_path)

# Extract features and target variable from training data
X_train = [[row['x1'], row['x2'], row['x3'], row['x4'], row['x5'], row['x6'], row['x7'], row['x8'], row['x9'], row['x10'], row['x11'], row['x12'], row['x13']] for row in train_data]
y_train = [row['y'] for row in train_data]

# Splitting training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Initialize SVM classifier
svm_model = SVC()

# Train SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)