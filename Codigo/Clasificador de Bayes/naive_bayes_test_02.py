import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

data = pd.read_csv(r'../datos_preprocesados.csv')
keys = list(data.head())

le = preprocessing.LabelEncoder()
y = le.fit_transform(data['0'].values)
data = data.drop(['0'], axis=1)
X = np.array(data.values, "float32")

#train
clf = GaussianNB()
clf.fit(X, y)

# predictions test
test_data = pd.read_csv(r'../test-02.csv')
y2 = test_data['0'].values
test_data = test_data.drop(['0'], axis=1)
X2 = np.array(test_data.values, "float32")
y2 = le.transform(y2)

predictions = clf.predict(X2)
print("Test 02 - Naive Bayes\n" + str(classification_report(y2,predictions))+ "_______________________________\n\n")

"""
predict = le.inverse_transform(predictions)
y2 = le.inverse_transform(y2)
print("real || predict")
for i in range(len(predictions)):
    print(y2.item(i) + ' , ' + str(predict.item(i)))
"""