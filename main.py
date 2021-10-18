import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
df = pd.read_csv('dados_gastos_propaganda.csv')
# print (df.head(12))
X = df.drop('vendas', axis=1)
y = df['vendas']
# print (X)
# print (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


model = Ridge(alpha=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print (mean_squared_error(y_test, y_pred))

#Executar o modelo utilizando alpha = [1, 1.5, 2, 2.5, ... 100]
#Ao final, mostrar o melhor valor de alpha, ou seja, aquele que der origem ao menor erro
min = mean_squared_error(y_test, y_pred)
alpha = 100
for i in np.linspace(1, 100, 199):
    model = Ridge(alpha=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    res = mean_squared_error(y_pred, y_test)
    if res < min:
        min = res
        alpha = i
print (f'O valor alpha={alpha} trouxe erro igual a {min}')
