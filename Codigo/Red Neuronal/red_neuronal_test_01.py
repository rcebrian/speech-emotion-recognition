import pandas as pd#ficheros
import numpy as np
from sklearn.metrics import classification_report#Report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing#Para preprocesar los datos (alegria, tristeza,etc)

#____________________________Importamos y balancemaos___________________________________
data = pd.read_csv(r'../datos_preprocesados.csv');
keys = list(data.head());

le = preprocessing.LabelEncoder()
y = le.fit_transform(data['0'].values)
data = data.drop(['0'], axis=1)
X = np.array(data.values, "float32")

#____________________________Entrenamos el modelo (Prueba diferentes parámetros)___________________________________
parameters = {'solver':['sgd','adam'], 'max_iter':[500,1000,1500,2000,2500], 'hidden_layer_sizes':[(400), (100,100,100,100), (100,100,100), (50,100,150), (200,100,50), (100, 80, 40, 40)]}
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
clf.fit(X,y)
#____________________________Importamos y preparamos datos de test__________________________________
data_test = pd.read_csv(r'../test-01.csv')
y_test = le.transform(data_test['0'].values)
data_test = data_test.drop(['0'], axis=1)
x_test = np.array(data_test.values, "float32")

#____________________________Predecimos y sacamos el report de los datos de test___________________________________
y_pred = clf.predict(x_test)
print("Test 01 - Red Neuronal\n" + str(classification_report(y_test,y_pred))+ "_______________________________\n\n")
