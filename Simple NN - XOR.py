#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Input data for XOR
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

# Network architecture
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(output_size)

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Backpropagation
    # Output layer error and gradient
    output_error = y - final_output
    output_gradient = sigmoid_derivative(final_output)
    output_delta = output_error * output_gradient

    # Hidden layer error and gradient
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_gradient = sigmoid_derivative(hidden_output)
    hidden_delta = hidden_error * hidden_gradient

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0) * learning_rate
    weights_input_hidden += x.T.dot(hidden_delta) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    # Calculate and print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = mean_squared_error_loss(y, final_output)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions after training
print("Final output after training:")
print(final_output)


# FEED FORWARD AND BACK PROPAAGTION

# In[19]:


import numpy as np

#Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Define means squared error loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

#Define the inputs
x=[1,0]
y=1
w1=np.array([[0.1,0.2],
            [0.3,0.4]])
w2=np.array([[0.5],
            [0.6]])
b1=np.array([[0.1],
            [0.2]])
b2=0.3

#Calculate the outputs
a1=np.dot(w1,x)+b1
h1=sigmoid(a1)
a2=np.dot(w2.T,a1)+b2
h2=sigmoid(a2)

#Print the outputs
print(a1)
print(a2)
print(mse_loss(y,a2))
print(sigmoid_derivative(a2))
print(sigmoid_derivative(a1))
print(np.dot(w2.T,sigmoid_derivative(a1)))
print(np.dot(w2.T,sigmoid_derivative(a1))*sigmoid_derivative(a2))
print(np.dot(w2.T,sigmoid_derivative(a1))*sigmoid_derivative(a2)*a1)

