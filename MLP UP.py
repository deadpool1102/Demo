# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the Breast Cancer dataset
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier
# MLPClassifier model with ReLU
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation=('relu'), max_iter=1000, random_state=42)
# Train the model on your data
mlp.fit(X_train, y_train)


# Train the model on the training data
mlp.fit(X_train, y_train)
# predictions on the test data
y_pred = mlp.predict(X_test)


# accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
