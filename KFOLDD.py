from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/infiniaclub/NeuralNetworkDataset/main/job.csv")

# Separate features and target
X = data.drop(columns=['hire'])
y = data['hire']

# Initialize network
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam',
                    alpha=1e-5, learning_rate_init=0.001, max_iter=1000)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#store accuracy
accuracy_scores = []

# Perform KFold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Access rows using iloc
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # Access rows using iloc
    # Train MLP classifier
    mlp.fit(X_train, y_train)
    # Predict on the test set
    y_pred = mlp.predict(X_test)
    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Print accuracy scores for each fold
for i, accuracy in enumerate(accuracy_scores):
    print(f"Fold {i+1} Accuracy: {accuracy}")

# average accuracy
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage Accuracy: {average_accuracy}")
