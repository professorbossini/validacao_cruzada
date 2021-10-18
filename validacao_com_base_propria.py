import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dados_gastos_propaganda.csv')

X = df.drop('vendas', axis=1)
y = df['vendas']

# usando a train_test_split obtenha as três bases: treinamento, validação e teste
# 70% para treinamento, 15% para validação e 15% para teste
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)

# print (len(X_train), len(X_validation), len(X_test) )

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

alpha = 100
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
y_validation_pred = model.predict(X_validation)
min = mean_squared_error(y_validation_pred, y_validation)
for i in np.linspace(1, 100, 199):
    model = Ridge(alpha=i)
    model.fit(X_train, y_train)
    y_validation_pred = model.predict(X_validation)
    res = mean_squared_error(y_validation_pred, y_validation)
    if res < min:
        min = res
        alpha = i
print (f'O valor alpha={alpha} trouxe erro igual a {min}')
