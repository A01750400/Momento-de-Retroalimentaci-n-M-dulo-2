#Importamos las librerias necesarias
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from random import randint

#Cargamos los datos ya preparados de un csv
df = pd.read_csv("automobile.csv")
print(df.head())
#Separamos los datos de entrada y salida en dataset distintos
X = df.drop(columns=['price'])
print(X.head())
y = df[['price']]
print(y.head())

#Normalizamos los datos de salida
y=(y - y.mean()) / y.std()

#Creamos una instancia del modelo
lr = LinearRegression()

#Separamos los datos entre datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Entrenamos al modelo
lr.fit(X_train, y_train)

#Hacemos una predicción
y_pred = lr.predict(X_test)

#Imprimimos el score del modelo
print("score using random_state =" + str(42) + "\n", lr.score(X_test, y_test))

#Hagamos predicciones con distintos valores de random state
best_prediction = 0
best_rand_val = 0
for i in range(100):
    random_state_val = randint(1, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state_val)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    if (score > best_prediction):
        best_prediction = score
        best_rand_val = random_state_val

    print("\nscore using random_state =" + str(random_state_val) + "\n", lr.score(X_test, y_test))

#Imprimimos la mejor predicción obtenida
print("\n\nbest prediciton was made using a random_state =", best_rand_val, "\nwith a score of: ", best_prediction)
