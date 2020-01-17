import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

data = pd.read_csv(r'../../prueba.csv')
keys = list(data.head())

le = preprocessing.LabelEncoder()
y = le.fit_transform(data['0'].values)
data = data.drop(['0'], axis=1)
X = np.array(data.values, "float32")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#train
clf = GaussianNB()
clf.fit(X_train, y_train)

# predictions validation
y_pred = clf.predict(X_test)
print("\nValidación - Naive Bayes\n" + str(classification_report(y_test,y_pred))+ "_______________________________\n\n")
