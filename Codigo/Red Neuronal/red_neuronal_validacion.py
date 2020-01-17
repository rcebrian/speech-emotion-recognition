import pandas as pd#ficheros
import numpy as np
from sklearn.metrics import classification_report#Report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing#Para preprocesar los datos (alegria, tristeza,etc)

#____________________________Importamos y balancemaos___________________________________
data = pd.read_csv(r'../datos_preprocesados.csv');
keys = list(data.head());

le = preprocessing.LabelEncoder()
y_data = le.fit_transform(data['0'].values)
data = data.drop(['0'], axis=1)
x_data = np.array(data.values, "float32")

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

#____________________________Entrenamos el modelo (Prueba diferentes parámetros)___________________________________
#parameters = {'solver': ['lbfgs','sgd','adam'], 'max_iter': [500,1000,1500,2000,2500], 'hidden_layer_sizes':[(400), (20,20,20,20), (100,100,100,100), (100,100,100), (50,100,150), (200,100,50), (100, 80, 40, 40)]}
parameters = {'hidden_layer_sizes': [(500)], 'max_iter': [1500], 'solver': ['adam']}
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
clf.fit(X_train,y_train)
print(clf.best_params_)#mejores parametros a la hora del entrenamiento

#____________________________Predecimos y sacamos el report de los datos de validación___________________________________
y_pred = clf.predict(X_test)
print("\nValidación - Red Neuronal\n" + str(classification_report(y_test,y_pred))+ "_______________________________\n\n")
print(le.inverse_transform(y_test))