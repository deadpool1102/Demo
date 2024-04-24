from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)

cv = KFold(n_splits=10, random_state=1, shuffle=True)

model = LogisticRegression()

scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))



from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
 

def get_dataset(n_samples=100):
 X, y = make_classification(n_samples=n_samples, n_features=7, n_informative=5, n_redundant=2, random_state=1)
 return X, y
 

def get_model():
    model = LogisticRegression()
    return model
 

def evaluate_model(cv):

    X, y = get_dataset()
 
    model = get_model()
 
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
 
    return mean(scores), scores.min(), scores.max()
 

ideal, _, _ = evaluate_model(LeaveOneOut())
print('Ideal: %.3f' % ideal)

folds = range(2,52)

means, mins, maxs = list(),list(),list()

for k in folds:
 
 cv = KFold(n_splits=k, shuffle=True, random_state=1)
 
 k_mean, k_min, k_max = evaluate_model(cv)
 
 print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
 
 means.append(k_mean)
 
 mins.append(k_mean - k_min)
 maxs.append(k_max - k_mean)

pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')

pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')

pyplot.show()
