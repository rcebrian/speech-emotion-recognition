import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

data = pd.read_csv('prueba.csv')
keys = list(data.head())

le = preprocessing.LabelEncoder()
# y_train = le.fit_transform(data['0'].values)

y = le.fit_transform(data['0'].values)
data = data.drop(['0'], axis=1)
X = np.array(data.values, "float32")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print(classification_report(y_test, y_pred))

# predictions
test_data = pd.read_csv('test-01.csv')
y2 = test_data['0'].values
test_data = test_data.drop(['0'], axis=1)
X2 = np.array(test_data.values, "float32")

predictions = clf.predict(X2)
predict = le.inverse_transform(predictions)

print("real || predict")
for i in range(len(predictions)):
    print(y2.item(i) + ' , ' + str(predict.item(i)))
