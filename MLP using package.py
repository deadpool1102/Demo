import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('diabetes.csv')

x=df.drop('Outcome',axis=1)
y=df.Outcome
x.head()
y.head()


X_test,X_train,y_test,y_train=train_test_split(x,y,test_size=0.2)
model=Sequential()
model.add(Dense(12,input_shape=(8,),activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=40,batch_size=10)

y_pred=model.predict(X_test)
y_pred_binary=np.where(y_pred>0.5,1,0)
accuracy1=accuracy_score(y_test,y_pred_binary)
print(accuracy1)


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("")

X = df.drop("Outcome", axis =1)
Y = df.Outcome
X.head()
Y.head()

model = Sequential()
model.add(Dense())
