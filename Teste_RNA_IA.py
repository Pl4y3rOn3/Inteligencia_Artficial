import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, \
    mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
import glob


# df = pd.read_csv('dados.csv')
# print(df.columns)

# collunsSTR = ['Ano', 'Região', 'UF', 'Código do Município', 'Nome do Município', 'Código da Escola',
#                     'Nome da Escola', 'Localização', 'Dependência Administrativa']

# print(df.dtypes)

# collunsUsed = ['Ano', 'Código do Município', 'Nome do Município', 'Código da Escola', 'Nome da Escola', 'Localização',
#                'Dependência Administrativa', 'Total Ensino Infantil', 'Creche', 'Pré-Escola', 'Total Ensino Fundamental',
#                'Anos Iniciais', 'Anos Finais', '1º Ano', '2° ano', '3° ano', '4° ano', '5° ano', '6° ano', '7° ano',
#                '8° ano', '9° ano', 'Turmas Multietapa', 'Total Ensino Medio', '1ª série', '2ª série', '3ª série', '4ª série',
#                'Não-Seriado']


def matrix_correlacao():
    df = pd.read_csv('dados.csv')

    le = preprocessing.LabelEncoder()

    collunsSTR = ['Ano', 'Código do Município', 'Nome do Município', 'Código da Escola',
                  'Nome da Escola', 'Localização', 'Dependência Administrativa']

    df[collunsSTR] = df[collunsSTR].apply(le.fit_transform)

    X = df[collunsSTR]
    # X = df[df.columns.difference(collunsSTR)]

    x1 = X.corr()

    sns.heatmap(x1, annot=True, vmin=-1, vmax=1)
    plt.show()


def previsao_alunos_infantil():
    df = pd.read_csv('dados.csv')

    collunsUsed = ['Ano', 'Nome do Município', 'Código da Escola', 'Nome da Escola', 'Localização',
                   'Dependência Administrativa', 'Creche', 'Pré-Escola', 'Total Ensino Fundamental',
                   'Anos Iniciais', 'Anos Finais', '1º Ano', '2° ano', '3° ano', '4° ano', '5° ano', '6° ano', '7° ano',
                   '8° ano', '9° ano', 'Turmas Multietapa', 'Total Ensino Medio', '1ª série', '2ª série', '3ª série', '4ª série',
                   'Não-Seriado']

    le = preprocessing.LabelEncoder()

    collunsSTR = ['Ano', 'Código do Município', 'Nome do Município', 'Código da Escola',
                  'Nome da Escola', 'Localização', 'Dependência Administrativa']

    df[collunsSTR] = df[collunsSTR].apply(le.fit_transform)

    collunChoice = 'Total Ensino Infantil'

    X = df[collunsUsed]
    y = df[collunChoice]

    # Separando modelo de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pre-processando
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural Network Treinando metodo
    mlp = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, activation='relu',
                       learning_rate='constant', early_stopping=True, verbose=False, tol=0.00001, n_iter_no_change=20)
    reg = mlp.fit(X_train_scaled, y_train.ravel())

    # Prevendo
    predictions = reg.predict(X_test_scaled)

    # Validando
    MSE = mean_squared_error(y_test, predictions, squared=True)
    RMSE = mean_squared_error(y_test, predictions, squared=False)
    cscore = r2_score(y_test, predictions)
    print("MSE = ", MSE)
    print("RMSE = ", RMSE)
    print("Score = ", cscore)
    # return MSE, cscore, RMSE

    # Plotando
    ref = np.linspace(min(y_test), max(y_test), 100)
    plt.scatter(y_test, predictions, color='crimson', s=10)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.plot(ref, ref, 'y')
    plt.show()


def previsao_alunos_fundamental():
    df = pd.read_csv('dados.csv')

    collunsUsed = ['Ano', 'Nome do Município', 'Código da Escola', 'Nome da Escola', 'Localização',
                   'Dependência Administrativa', 'Total Ensino Infantil', 'Creche', 'Pré-Escola',
                   'Anos Iniciais', 'Anos Finais', '1º Ano', '2° ano', '3° ano', '4° ano', '5° ano', '6° ano', '7° ano',
                   '8° ano', '9° ano', 'Turmas Multietapa', 'Total Ensino Medio', '1ª série', '2ª série', '3ª série', '4ª série',
                   'Não-Seriado']

    le = preprocessing.LabelEncoder()

    collunsSTR = ['Ano', 'Código do Município', 'Nome do Município', 'Código da Escola',
                  'Nome da Escola', 'Localização', 'Dependência Administrativa']

    df[collunsSTR] = df[collunsSTR].apply(le.fit_transform)

    collunChoice = 'Total Ensino Fundamental'

    X = df[collunsUsed]
    y = df[collunChoice]

    # Separando modelo de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pre-processando
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural Network Treinando metodo
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000, activation='relu',
                       learning_rate='constant', early_stopping=True, verbose=False, tol=0.000001, n_iter_no_change=20)
    reg = mlp.fit(X_train_scaled, y_train.ravel())

    # Prevendo
    predictions = reg.predict(X_test_scaled)

    # Validando
    MSE = mean_squared_error(y_test, predictions, squared=True)
    RMSE = mean_squared_error(y_test, predictions, squared=False)
    cscore = r2_score(y_test, predictions)
    print("MSE = ", MSE)
    print("RMSE = ", RMSE)
    print("Score = ", cscore)
    # return MSE, cscore, RMSE

    # Plotando
    ref = np.linspace(min(y_test), max(y_test), 100)
    plt.scatter(y_test, predictions, color='crimson', s=10)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.plot(ref, ref, 'y')
    plt.show()


def previsao_alunos_medio():
    df = pd.read_csv('dados.csv')

    collunsUsed = ['Ano', 'Nome do Município', 'Código da Escola', 'Nome da Escola', 'Localização',
                   'Dependência Administrativa', 'Total Ensino Infantil', 'Creche', 'Pré-Escola', 'Total Ensino Fundamental',
                   'Anos Iniciais', 'Anos Finais', '1º Ano', '2° ano', '3° ano', '4° ano', '5° ano', '6° ano', '7° ano',
                   '8° ano', '9° ano', 'Turmas Multietapa', '1ª série', '2ª série', '3ª série', '4ª série',
                   'Não-Seriado']

    le = preprocessing.LabelEncoder()

    collunsSTR = ['Ano', 'Código do Município', 'Nome do Município', 'Código da Escola',
                  'Nome da Escola', 'Localização', 'Dependência Administrativa']

    df[collunsSTR] = df[collunsSTR].apply(le.fit_transform)

    collunChoice = 'Total Ensino Medio'

    X = df[collunsUsed]
    y = df[collunChoice]

    # Separando modelo de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pre-processando
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural Network Treinando metodo
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000, activation='relu',
                       learning_rate='constant', early_stopping=True, verbose=False, tol=0.000001, n_iter_no_change=20)
    reg = mlp.fit(X_train_scaled, y_train.ravel())

    # Prevendo
    predictions = reg.predict(X_test_scaled)

    # Validando
    MSE = mean_squared_error(y_test, predictions, squared=True)
    RMSE = mean_squared_error(y_test, predictions, squared=False)
    cscore = r2_score(y_test, predictions)
    print("MSE = ", MSE)
    print("RMSE = ", RMSE)
    print("Score = ", cscore)
    # return MSE, cscore, RMSE

    # Plotando
    ref = np.linspace(min(y_test), max(y_test), 100)
    plt.scatter(y_test, predictions, color='crimson', s=10)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.plot(ref, ref, 'y')
    plt.show()


def previsao_total_alunos():
    # Carregando bancos de dados
    df = pd.read_csv('dados_teste.csv')
    df1 = pd.read_csv('dados_real.csv')

    # Selecionando colunas que vão ser usadas como entrada
    collunsUsed = ['Ano', 'Nome do Município', 'Código da Escola', 'Nome da Escola', 'Localização',
                   'Dependência Administrativa', 'Total Ensino Infantil', 'Creche', 'Pré-Escola', 'Total Ensino Fundamental',
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
    collunChoice = 'TotalAlunosEscola'

    # Selecionando conjunto de dados de Treino para Teste
    X = df[collunsUsed]
    y = df[collunChoice]

    # Selecionando conjunto de dados Reais para validação
    X_real = df1[collunsUsed]
    y_real = df1[collunChoice]

    # Separando modelo de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Pre-processando
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Normalizando dados de Teste
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Normalizando dados Reais
    X_real_scaled = scaler.transform(X_real)

    # Definindo Predefinições do modelo de Artificial Neural Network
    mlp = MLPRegressor(hidden_layer_sizes=(5,5,5), max_iter=1000, activation='relu',
                       learning_rate='constant', early_stopping=True, verbose=False, tol=0.00001, n_iter_no_change=20)
    
    
    # Treinando modelo de RNA baseado no X e y de Treino
    reg = mlp.fit(X_train_scaled, y_train.ravel())

    # Prevendo
    predictions = reg.predict(X_test_scaled)
    predictions_real = reg.predict(X_real_scaled)

    # Validando casos de Teste
    MSE = mean_squared_error(y_test, predictions, squared=True)
    RMSE = mean_squared_error(y_test, predictions, squared=False)
    cscore = r2_score(y_test, predictions)

    # Validando casos Reais
    MSE_r = mean_squared_error(y_real, predictions_real, squared=True)
    RMSE_r = mean_squared_error(y_real, predictions_real, squared=False)
    cscore_r = r2_score(y_real, predictions_real)

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
    
    # Plotando
    ref = np.linspace(min(y_test), max(y_test), 100)
    plt.scatter(y_test, predictions, color='crimson', s=10)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.plot(ref, ref, 'y')
    plt.show()

    ref = np.linspace(min(y_real), max(y_real), 100)
    plt.scatter(y_real, predictions_real, color='crimson', s=10)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.plot(ref, ref, 'y')
    plt.show()




inicio = time.time()
# previsao_alunos_infantil()
# previsao_alunos_fundamental()
# previsao_alunos_medio()
previsao_total_alunos()
fim = time.time()

print("Tempo de execução (S):", fim-inicio)