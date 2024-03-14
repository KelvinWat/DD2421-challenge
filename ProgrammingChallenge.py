import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
data = pd.read_csv('TrainOnMe_orig.csv')

# Convert x7 to numerical using Label Encoding
label_encoder = LabelEncoder()
data['x7'] = label_encoder.fit_transform(data['x7'])

# Encode string labels in y using Label Encoding
data['y'] = label_encoder.fit_transform(data['y'])

# Drop the unnecessary column
data.drop('Unnamed: 0', axis=1, inplace=True)

# Split the data into features (X) and target variable (y)
X = data.drop(['y'], axis=1)
y = data['y']

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Define the neural network model for multi-class classification with dropout
def create_model(X_train_scaled):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),  # Dropout layer with 20% dropout rate
        Dense(32, activation='relu'),
        Dropout(0.2),  # Dropout layer with 20% dropout rate
        Dense(len(label_encoder.classes_), activation='softmax')  # Use softmax activation for multi-class classification
    ])
    return model

# Perform 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in skf.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)

    model = create_model(X_train_scaled)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_scaled, y_train_fold, epochs=50, batch_size=64, verbose=0)

    # Evaluate the model
    _, accuracy = model.evaluate(X_val_scaled, y_val_fold)
    cv_scores.append(accuracy)

# Calculate average cross-validation accuracy
average_cv_accuracy = sum(cv_scores) / len(cv_scores)
print("Average Cross-Validation Accuracy:", average_cv_accuracy)

# Re-train the model on the entire training data
X_train_scaled = scaler.fit_transform(X)
model = create_model(X_train_scaled)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y, epochs=50, batch_size=64, verbose=1)

# Load the evaluation dataset
eval_data = pd.read_csv('EvaluateOnMe.csv')

# Convert x7 to numerical using Label Encoding
eval_data['x7'] = label_encoder.fit_transform(eval_data['x7'])  # Reuse the same label encoder used for training data

# Drop the unnecessary column
eval_data.drop('Unnamed: 0', axis=1, inplace=True)

# Standardize features by removing the mean and scaling to unit variance
eval_data_scaled = scaler.transform(eval_data)

# Make predictions using the trained model
y_pred_eval = model.predict(eval_data_scaled)

# Convert probabilities to class labels
y_pred_eval_classes = y_pred_eval.argmax(axis=-1)

# Write the predictions to result.txt
with open('result.txt', 'w') as f:
    for pred in y_pred_eval_classes:
        f.write(str(pred) + '\n')


