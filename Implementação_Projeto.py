from numpy import mean
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors

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

# Definindo o número de folds e repetições
num_folds = 10
num_repeats = 5

# Configurando inicializador de regressor linear
regressor = LinearRegression()

# Configurando inicializador de rede neural artificial 
mlp = MLPRegressor(hidden_layer_sizes=(5,5,5), max_iter=1000, activation='relu',
                       learning_rate='constant', early_stopping = True,verbose= False, tol= 0.000001, n_iter_no_change= 20)

# Configurando inicializadores de KNN para distancias de 1, 7 e 15
model1 = neighbors.KNeighborsRegressor(n_neighbors = 1, weights= 'distance')
model2 = neighbors.KNeighborsRegressor(n_neighbors = 7, weights= 'distance')
model3 = neighbors.KNeighborsRegressor(n_neighbors = 15, weights= 'distance')

# Criando Listas para pegar valores médidos de MSE, RMSE, R-Score e Tempo da Regressão Linear
RL_mse_teste = []
RL_mse_real = []
RL_rmse_teste = []
RL_rmse_real = []
RL_score_teste = []
RL_score_real = []
RL_tempo = []

# Criando Listas para pegar valores médidos de MSE, RMSE, R-Score e Tempo da Rede Neural Artificial
RNA_mse_teste = []
RNA_mse_real = []
RNA_rmse_teste = []
RNA_rmse_real = []
RNA_score_teste = []
RNA_score_real = []
RNA_tempo = []

# Criando Listas para pegar valores médidos de MSE, RMSE, R-Score e Tempo do KNN com k = 1
KNN1_mse_teste = []
KNN1_mse_real = []
KNN1_rmse_teste = []
KNN1_rmse_real = []
KNN1_score_teste = []
KNN1_score_real = []
KNN1_tempo = []

# Criando Listas para pegar valores médidos de MSE, RMSE, R-Score e Tempo do KNN com k = 7
KNN7_mse_teste = []
KNN7_mse_real = []
KNN7_rmse_teste = []
KNN7_rmse_real = []
KNN7_score_teste = []
KNN7_score_real = []
KNN7_tempo = []

# Criando Listas para pegar valores médidos de MSE, RMSE, R-Score e Tempo do KNN com k = 15
KNN15_mse_teste = []
KNN15_mse_real = []
KNN15_rmse_teste = []
KNN15_rmse_real = []
KNN15_score_teste = []
KNN15_score_real = []
KNN15_tempo = []

# Looping para repetir o processo de validação cruzada 5 vezes
for repeat in range(num_repeats):
    print(f"Repetition {repeat + 1}")

    # Criando objetos KFold para dividir os dados
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat)

    # Inicializando Timer para RL
    inicio = time.time()

    # Loop para executar a validação cruzada na Regressão Linear
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        
        # Separando casos de Treino e casos de Teste
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Pre-processando
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Normalizando dados de Teste
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Normalizando dados Reais
        X_real_scaled = scaler.transform(X_real)

        # Treinando modelo de RL baseado no X e y de Treino
        regressor.fit(X_train_scaled, y_train)
        
        # Realizando Predições
        y_pred = regressor.predict(X_test_scaled)
        ypred_real = regressor.predict(X_real_scaled)
        
        # Validando casos de Teste
        MSE = mean_squared_error(y_test, y_pred, squared=True)
        RMSE = mean_squared_error(y_test, y_pred, squared=False)
        cscore = r2_score(y_test, y_pred)

        # Validando casos Reais
        MSE_r = mean_squared_error(y_real, ypred_real, squared=True)
        RMSE_r = mean_squared_error(y_real, ypred_real, squared=False)
        cscore_r = r2_score(y_real, ypred_real)

        # Adicionando valores a lista criada anteriormente para poder pegar a média
        RL_mse_teste.append(MSE)
        RL_mse_real.append(MSE_r)
        RL_rmse_teste.append(RMSE)
        RL_rmse_real.append(RMSE_r)
        RL_score_teste.append(cscore)
        RL_score_real.append(cscore_r)
        
        # print dos valores preditos e reais para teste
        # teste1 = pd.DataFrame({"real": y_real, "predito": ypred_real.tolist()})
        # print(teste1.head(20))
    
    # Finalizando Timer e Adicionando a Lista de Tempo da RL
    fim = time.time()
    RL_tempo.append((fim - inicio))

    # Inicializando Timer para RNA
    inicio = time.time()

    # Loop para executar a validação cruzada da rede neural artificial
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        
        # Separando casos de Treino e casos de Teste
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Pre-processando
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Normalizando dados de Teste
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Normalizando dados Reais
        X_real_scaled = scaler.transform(X_real)
                
        # Treinando modelo de RNA baseado no X e y de Treino
        reg = mlp.fit(X_train_scaled, y_train.ravel())
        
        # Realizando Predições
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

        # Adicionando valores a lista criada anteriormente para poder pegar a média
        RNA_mse_teste.append(MSE)
        RNA_mse_real.append(MSE_r)
        RNA_rmse_teste.append(RMSE)
        RNA_rmse_real.append(RMSE_r)
        RNA_score_teste.append(cscore)
        RNA_score_real.append(cscore_r)
        
        # print dos valores preditos e reais para teste
        # teste2 = pd.DataFrame({"real": y_real, "predito": predictions_real.tolist()})
        # print(teste2.head(20))

    # Finalizando Timer e Adicionando a Lista de Tempo do RNA
    fim = time.time()
    RNA_tempo.append((fim - inicio))

    # Inicializando Timer para KNN com K = 1
    inicio = time.time()

    # Loop para executar a validação cruzada da KNN com k = 1
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        
        # Separando casos de Treino e casos de Teste
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Pre-processando
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Normalizando dados de Teste
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Normalizando dados Reais
        X_real_scaled = scaler.transform(X_real)
        
        # Treinando modelo de KNN baseado no X e y de Treino
        model1.fit(X_train_scaled, y_train)

        # Realizando Predições
        pred = model1.predict(X_test_scaled)
        pred_real = model1.predict(X_real_scaled)

        # Validando casos de Teste
        MSE = mean_squared_error(y_test, pred, squared=True)
        RMSE = mean_squared_error(y_test, pred, squared=False)
        cscore = r2_score(y_test, pred)

        # Validando casos Reais
        MSE_r = mean_squared_error(y_real, pred_real, squared=True)
        RMSE_r = mean_squared_error(y_real, pred_real, squared=False)
        cscore_r = r2_score(y_real, pred_real)

        # Adicionando valores a lista criada anteriormente para poder pegar a média
        KNN1_mse_teste.append(MSE)
        KNN1_mse_real.append(MSE_r)
        KNN1_rmse_teste.append(RMSE)
        KNN1_rmse_real.append(RMSE_r)
        KNN1_score_teste.append(cscore)
        KNN1_score_real.append(cscore_r)

        # print dos valores preditos e reais para teste        
        # teste3 = pd.DataFrame({"real": y_real, "predito": pred_real.tolist()})
        # print(teste3.head(20))

    # Finalizando Timer e Adicionando a Lista de Tempo do KNN com k = 1
    fim = time.time()
    KNN1_tempo.append((fim - inicio))

    # Inicializando Timer para KNN com K = 7
    inicio = time.time()

    # Loop para executar a validação cruzada da KNN com k = 7
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        
        # Separando casos de Treino e casos de Teste
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Pre-processando
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Normalizando dados de Teste
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Normalizando dados Reais
        X_real_scaled = scaler.transform(X_real)
        
        # Treinando modelo de KNN baseado no X e y de Treino
        model2.fit(X_train_scaled, y_train)

        # Realizando Predições
        pred = model2.predict(X_test_scaled)
        pred_real = model2.predict(X_real_scaled)

        # Validando casos de Teste
        MSE = mean_squared_error(y_test, pred, squared=True)
        RMSE = mean_squared_error(y_test, pred, squared=False)
        cscore = r2_score(y_test, pred)

        # Validando casos Reais
        MSE_r = mean_squared_error(y_real, pred_real, squared=True)
        RMSE_r = mean_squared_error(y_real, pred_real, squared=False)
        cscore_r = r2_score(y_real, pred_real)

        # Adicionando valores a lista criada anteriormente para poder pegar a média
        KNN7_mse_teste.append(MSE)
        KNN7_mse_real.append(MSE_r)
        KNN7_rmse_teste.append(RMSE)
        KNN7_rmse_real.append(RMSE_r)
        KNN7_score_teste.append(cscore)
        KNN7_score_real.append(cscore_r)
        
        # print dos valores preditos e reais para teste        
        # teste4 = pd.DataFrame({"real": y_real, "predito": pred_real.tolist()})
        # print(teste4.head(20))

    # Finalizando Timer e Adicionando a Lista de Tempo do KNN com k = 7
    fim = time.time()
    KNN7_tempo.append((fim - inicio))

    # Inicializando Timer para KNN com K = 15
    inicio = time.time()

    # Loop para executar a validação cruzada da KNN com k = 15
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        
        # Separando casos de Treino e casos de Teste
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Pre-processando
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Normalizando dados de Teste
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Normalizando dados Reais
        X_real_scaled = scaler.transform(X_real)
        
        # Treinando modelo de KNN baseado no X e y de Treino
        model3.fit(X_train_scaled, y_train)

        # Realizando Predições
        pred = model3.predict(X_test_scaled)
        pred_real = model3.predict(X_real_scaled)

        # Validando casos de Teste
        MSE = mean_squared_error(y_test, pred, squared=True)
        RMSE = mean_squared_error(y_test, pred, squared=False)
        cscore = r2_score(y_test, pred)

        # Validando casos Reais
        MSE_r = mean_squared_error(y_real, pred_real, squared=True)
        RMSE_r = mean_squared_error(y_real, pred_real, squared=False)
        cscore_r = r2_score(y_real, pred_real)

        # Adicionando valores a lista criada anteriormente para poder pegar a média
        KNN15_mse_teste.append(MSE)
        KNN15_mse_real.append(MSE_r)
        KNN15_rmse_teste.append(RMSE)
        KNN15_rmse_real.append(RMSE_r)
        KNN15_score_teste.append(cscore)
        KNN15_score_real.append(cscore_r)
        
        # print dos valores preditos e reais para teste
        # teste5 = pd.DataFrame({"real": y_real, "predito": pred_real.tolist()})
        # print(teste5.head(20))

    # Finalizando Timer e Adicionando a Lista de Tempo do KNN com k = 15
    fim = time.time()
    KNN15_tempo.append((fim - inicio))


# Printando dados na tela
print("--------------Regressão-Linear------------------")
print("MSE Teste:", mean(RL_mse_teste))
print("RMSE Teste:", mean(RL_rmse_teste))
print("R-scores Teste:", mean(RL_score_teste))
print("------------------------------------------------")
print("MSE Real:", mean(RL_mse_real))
print("RMSE Real:", mean(RL_rmse_real))
print("R-scores Real:", mean(RL_score_real))
print("------------Rede-Neural-Artificial--------------")
print("MSE Teste:", mean(RNA_mse_teste))
print("RMSE Teste:", mean(RNA_rmse_teste))
print("R-scores Teste:", mean(RNA_score_teste))
print("------------------------------------------------")
print("MSE Real:", mean(RNA_mse_real))
print("RMSE Real:", mean(RNA_rmse_real))
print("R-scores Real:", mean(RNA_score_real))
print("--------------------KNN-k=1---------------------")
print("MSE Teste:", mean(KNN1_mse_teste))
print("RMSE Teste:", mean(KNN1_rmse_teste))
print("R-scores Teste:", mean(KNN1_score_teste))
print("------------------------------------------------")
print("MSE Real:", mean(KNN1_mse_real))
print("RMSE Real:", mean(KNN1_rmse_real))
print("R-scores Real:", mean(KNN1_score_real))
print("--------------------KNN-k=7---------------------")
print("MSE Teste:", mean(KNN7_mse_teste))
print("RMSE Teste:", mean(KNN7_rmse_teste))
print("R-scores Teste:", mean(KNN7_score_teste))
print("------------------------------------------------")
print("MSE Real:", mean(KNN7_mse_real))
print("RMSE Real:", mean(KNN7_rmse_real))
print("R-scores Real:", mean(KNN7_score_real))
print("--------------------KNN-k=15--------------------")
print("MSE Teste:", mean(KNN15_mse_teste))
print("RMSE Teste:", mean(KNN15_rmse_teste))
print("R-scores Teste:", mean(KNN15_score_teste))
print("------------------------------------------------")
print("MSE Real:", mean(KNN15_mse_real))
print("RMSE Real:", mean(KNN15_rmse_real))
print("R-scores Real:", mean(KNN15_score_real))
print("--------------Tempo-dos-Algoritmos--------------")
print("Tempo Total RL (s): ", sum(RL_tempo))
print("Tempo Total RNA (s): ", sum(RNA_tempo))
print("Tempo Total KNN1 (s): ", sum(KNN1_tempo))
print("Tempo Total KNN7 (s): ", sum(KNN7_tempo))
print("Tempo Total KNN15 (s): ", sum(KNN15_tempo))


