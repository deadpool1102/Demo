import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassfier
import matplotlib.pyplot as plt
import io
df=pd.read_csv('/content/drug200.csv')
df
d = {'HIGH': 0, 'NORMAL': 1}
df['Cholesterol'] = df['Cholesterol'].map(d)
d = {'HIGH': 0, 'LOW': 1, 'NORMAL': 2}
df['BP'] = df['BP'].map(d)
d = {'F': 0, 'M': 1}
df['Sex'] = df['Sex'].map(d)
features = ['Age','Sex','BP','Cholesterol','Na_to_K']
X = df[features]
y = df['Drug']
dtree = DecisionTreeClassifier().fit(X, y)
tree.plot_tree(dtree, feature_names=features)