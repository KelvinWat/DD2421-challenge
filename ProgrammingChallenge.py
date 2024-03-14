import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('TrainOnMe_orig.csv')

# Convert x7 to numerical using Label Encoding
label_encoder = LabelEncoder()
data['x7'] = label_encoder.fit_transform(data['x7'])

# Drop the unnecessary column
data.drop('Unnamed: 0', axis=1, inplace=True)

# Split the data into features (X) and target variable (y)
X = data.drop(['y'], axis=1)
y = data['y']

# Standardize features by removing the mean and scaling to unit variance
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# Create and train RandomForestClassifier with best parameters
best_rf_model = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=300)

# Perform cross-validation on the training set
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)  # 5-fold cross-validation

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Fit the model on the entire training set
best_rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_test = best_rf_model.predict(X_test)

# Calculate accuracy on the testing set
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Accuracy on Testing Set:", test_accuracy)

# Load the evaluation dataset
eval_data = pd.read_csv('EvaluateOnMe.csv')

# Convert x7 to numerical using Label Encoding (assuming same label encoder used for training data)
eval_data['x7'] = label_encoder.transform(eval_data['x7'])

# Drop the unnecessary column
eval_data.drop('Unnamed: 0', axis=1, inplace=True)

# Standardize features by removing the mean and scaling to unit variance (using the same scaler as training data)
eval_data_scaled = scaler.transform(eval_data)

# Make predictions using the trained model
y_pred_eval = best_rf_model.predict(eval_data_scaled)

# Write the predictions to result.txt
with open('result.txt', 'w') as f:
    for pred in y_pred_eval:
        f.write(str(pred) + '\n')

