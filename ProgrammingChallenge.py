import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)  # 5-fold cross-validation

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Train the model on the entire dataset
rf_model.fit(X_scaled, y)

# Optionally, you can evaluate the model on a holdout test set
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
# y_pred = rf_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy on Test Set:", accuracy)

# Load the evaluation dataset
eval_data = pd.read_csv('EvaluateOnMe.csv')

# Convert x7 to numerical using Label Encoding
eval_data['x7'] = label_encoder.transform(eval_data['x7'])

# Drop the unnecessary column
eval_data.drop('Unnamed: 0', axis=1, inplace=True)

# Standardize features by removing the mean and scaling to unit variance
eval_data_scaled = scaler.transform(eval_data)

# Make predictions using the trained model
y_pred_eval = rf_model.predict(eval_data_scaled)

# Write the predictions to result.txt
with open('result.txt', 'w') as f:
    for pred in y_pred_eval:
        f.write(str(pred) + '\n')
