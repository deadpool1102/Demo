#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 

  

def predict(row, weights): 

  activation = weights[0]  ## Storing Bias 

  for i in range(len(row)-1):  ## For Loop 

    activation += weights[i+1] * row[i] 

  return 1.0 if activation >=0 else 0 

  

def training_weights(train_data, learning_rate, n_epoch): 

  weights = [00 for i in range(len(train_data[0]))] 

  for epoch in range(n_epoch):          ## Loop for number of epoch 

    sum_error = 0.0 

    for row in train_data:       ## Loop for each row in training 

      prediction = predict(row, weights) 

      error = row[-1] - prediction 

      sum_error += error**2             ## Mean Square Error 

      weights[0] += learning_rate * error  ## Updating Bias 

      for i in range(len(row)-1):       ## Loop for each weight update for row 

        weights[i+1] += learning_rate * error * row[i] ## Updating Weights 

    print(f">epoch= {epoch}, learning_rate = {learning_rate}, MSE= {sum_error}") 

  return weights 

 

logical_and_dataset = pd.read_csv("https://raw.githubusercontent.com/infiniaclub/NeuralNetworkDataset/main/logical_and.csv").values 

learning_rate = 0.01  ## 1% Learning Rate 

n_epoch = 25    ## Change n_epoch or learning_rate to see effect on prediction 

weights = training_weights(logical_and_dataset, learning_rate, n_epoch) 

print(f"\n Computed Bias : {round(weights[0],3)} \n Computed Weights_i: {weights[1:]} \n") 

  

## Predictiion 

print("Prediction for AND Dataset") 

for row in logical_and_dataset: 
    prediction = predict(row, weights) 

print(f"Actual: {round(row[-1])}  Predicted: {round(prediction)}") 

  

## OR GATE 

or_gate = [[1,1,1], 

            [1,0,1], 

            [0,1,1], 

            [0,0,0]] 

learning_rate = 0.01 

n_epoch = 25 

weights = training_weights(or_gate, learning_rate, n_epoch) 

print(f"\nTrain Network for OR Gate\nComputed Bias : {round(weights[0],3)} \nComputed Weights: {weights[1:]} \n") 

## Predictiion 

print("Prediction for OR Dataset") 

for row in or_gate: 

    prediction = predict(row, weights) 

print(f"Actual: {round(row[-1])}  Predicted: {round(prediction)}") 

  

## XOR GATE 

xor_gate = [[1,1,0], 

            [1,0,1], 

            [0,1,1], 

            [0,0,0]] 

learning_rate = 0.01 

n_epoch = 10 

weights = training_weights(xor_gate, learning_rate, n_epoch) 

print(f"\nTrain Network for AND Gate\nComputed Bias : {round(weights[0],3)} \nComputed Weights: {weights[1:]} \n") 

## Predictiion 

print("Prediction for XOR Dataset") 

for row in xor_gate: 

    prediction = predict(row, weights) 

print(f"Actual: {round(row[-1])}  Predicted: {round(prediction)}") 


# In[ ]:




