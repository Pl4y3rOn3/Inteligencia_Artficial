import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import neighbors

# Carregando bancos de dados
df = pd.read_csv('dados_teste.csv')
df1 = pd.read_csv('dados_real.csv')

# Selecionando colunas que vão ser usadas como entrada
collunsUsed = ['Ano', 'Nome do Município', 'Código da Escola', 'Nome da Escola', 'Localização', 
            'Dependência Administrativa', 'Creche', 'Pré-Escola', 'Total Ensino Fundamental', 
            'Anos Iniciais', 'Anos Finais', '1º Ano', '2° ano', '3° ano', '4° ano', '5° ano', '6° ano', '7° ano', 
            '8° ano', '9° ano', 'Turmas Multietapa', 'Total Ensino Medio', '1ª série', '2ª série', '3ª série', '4ª série', 
            'Não-Seriado']

# Selecionando colunas de String para conversão
collunsSTR = ['Ano', 'Código do Município', 'Nome do Município', 'Código da Escola', 
                'Nome da Escola', 'Localização', 'Dependência Administrativa']

# Função de transformar categorias de Strings em numeros
le = preprocessing.LabelEncoder()

# Transformando colunas de Strings em numeros
df[collunsSTR] = df[collunsSTR].apply(le.fit_transform)
df1[collunsSTR] = df1[collunsSTR].apply(le.fit_transform)

# Escolha de coluna a ser predita
collunChoice = 'Total Ensino Infantil'

# Selecionando conjunto de dados de Treino para Teste
X = df[collunsUsed]
y = df[collunChoice]

# Selecionando conjunto de dados Reais para validação
X_real = df1[collunsUsed]
y_real = df1[collunChoice]

# Separando modelo de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)

# Pre-processando
scaler = StandardScaler()
scaler.fit(X_train)

# Normalizando dados de Teste
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalizando dados Reais
X_real_scaled = scaler.transform(X_real)

# Definindo predefinições do modelo de KNN
model = neighbors.KNeighborsRegressor(n_neighbors = 15, weights= 'distance')

# Treinando modelo de KNN baseado no X e y de Treino
model.fit(X_train_scaled, y_train)  #fit the model

# Prevendo
pred = model.predict(X_test_scaled) #make prediction on test set
pred_real = model.predict(X_real_scaled)

# Validando casos de Teste
MSE = mean_squared_error(y_test, pred, squared=True)
RMSE = mean_squared_error(y_test, pred, squared=False)
cscore = r2_score(y_test, pred)

# Validando casos Reais
MSE_r = mean_squared_error(y_real, pred_real, squared=True)
RMSE_r = mean_squared_error(y_real, pred_real, squared=False)
cscore_r = r2_score(y_real, pred_real)

# Printando casos de Teste
print("=======Dados=de=Teste=======")
print("MSE = ", MSE)
print("RMSE = ", RMSE)
print("Score = ", cscore)

# Printando casos Reais
print("=========Dados=Reais========")
print("MSE = ", MSE_r)
print("RMSE = ", RMSE_r)
print("Score = ", cscore_r)



def sePrecisar():
     # rmse_val = [] #to store rmse values for different k
    # for K in range(20):
    #     K = K+1
    #     model = neighbors.KNeighborsRegressor(n_neighbors = K)

    #     model.fit(X_train_scaled, y_train)  #fit the model
    #     pred = model.predict(X_test_scaled) #make prediction on test set

    #     pred_real = model.predict(X_real_scaled)
    #     MSE_r = mean_squared_error(y_real, pred_real)

    #     error = mean_squared_error(y_test,pred) #calculate rmse
    #     rmse_val.append(error) #store rmse values
    #     print('MSE value for k= ' , K , 'is:', error)
    #     print('MSE value for k real = ' , K , 'is:', MSE_r)


    # curve = pd.DataFrame(rmse_val) #elbow curve 
    # curve.plot()
    # plt.show()

    # k = [1,7,20]

    # for i in k:
    #     model = neighbors.KNeighborsRegressor(n_neighbors = i)
    #     model.fit(X_train_scaled, y_train)  #fit the model
    #     pred = model.predict(X_test_scaled) #make prediction on test set

    #     MSE = mean_squared_error(y_test, pred, squared=True)
    #     RMSE = mean_squared_error(y_test, pred, squared=False)
    #     cscore = r2_score(y_test, pred)
    #     print("MSE = ", MSE)
    #     print("RMSE = ", RMSE)
    #     print("Score = ",cscore)
    return