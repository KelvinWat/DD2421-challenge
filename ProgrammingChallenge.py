import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN model
k = 5  # Number of neighbors to consider
knn_model = KNeighborsClassifier(n_neighbors=k)

# Fit the model
knn_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Load the evaluation dataset
eval_data = pd.read_csv('EvaluateOnMe.csv')

# Convert x7 to numerical using Label Encoding
label_encoder = LabelEncoder()
eval_data['x7'] = label_encoder.fit_transform(eval_data['x7'])

# Drop the unnecessary column
eval_data.drop('Unnamed: 0', axis=1, inplace=True)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
eval_data_scaled = scaler.fit_transform(eval_data)

# Make predictions using the trained model (assuming it's already trained)
y_pred_eval = knn_model.predict(eval_data_scaled)  # Using the KNN model from the previous example

# Write the predictions to result.txt
with open('result.txt', 'w') as f:
    for pred in y_pred_eval:
        f.write(str(pred) + '\n')