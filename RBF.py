import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as numpy


Data= pd.read_csv("bank-full.csv")
cols= ["age","balance","day","duration","campaign","pdays","previous"]
data_encode= Data.drop(cols, axis= 1)
data_encode= data_encode.apply(LabelEncoder().fit_transform)
data_rest= Data[cols]
Data= pd.concat([data_rest,data_encode], axis= 1)


data_train, data_test = train_test_split(Data, test_size = 0.5, random_state = 4)
#data_train = Data
#data_test = Data
x_train = data_train.drop("y", axis = 1)
y_train = data_train["y"]
x_test = data_test.drop("y", axis = 1)
y_test = data_test["y"]
#x_train.head()

scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


#K_cent = 4
K_cent = 8
km = KMeans(n_clusters = K_cent, max_iter= 100)
km.fit(x_train)
cent = km.cluster_centers_


max = 0
for i in range(K_cent):
    for j in range(K_cent):
        d = numpy.linalg.norm(cent[i] - cent[j])
        if d > max:
            max = d

d = max
sigma = d / math.sqrt(2 * K_cent)
print(sigma)


shape = x_train.shape
row = shape[0]
column = K_cent
G = numpy.empty((row, column), dtype = float)
for i in range(row):
    for j in range(column):
        dist = numpy.linalg.norm(x_train[i] - cent[j])
        G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))
print(G)


GTG = numpy.dot(G.T, G)
GTG_inv = numpy.linalg.inv(GTG)
fac = numpy.dot(GTG_inv, G.T)
W = numpy.dot(fac, y_train)
print(W)


row = x_test.shape[0]
column = K_cent
G_test = numpy.empty((row, column), dtype = float)
for i in range(row):
    for j in range(column):
        #dist = numpy.linalg.norm(x_test.iloc[i] - cent[j])
        dist = numpy.linalg.norm(x_test[i] - cent[j])
        G_test[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))
print(G_test[0])


prediction = numpy.dot(G_test, W)
prediction = 0.5 * (numpy.sign(prediction - 0.5) + 1)
score = accuracy_score(prediction, y_test)
print(score.mean())
