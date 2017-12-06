from sklearn.naive_bayes import MultinomialNB
import pandas as pd 
import numpy as np 

data = pd.read_csv('../data/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
y = data[:, -1]

X_train = X[:-100,]
y_train = y[:-100,]
X_test = X[-100:,]
y_test = y[-100:,]


clf = MultinomialNB()
clf.fit(X_train, y_train)
s = clf.score(X_test, y_test)
print("score Naive Bayes: ", s)


from sklearn.ensemble import AdaBoostClassifier
mdl = AdaBoostClassifier()
mdl.fit(X_train, y_train)
print("score Adaboost: ", mdl.score(X_test, y_test))