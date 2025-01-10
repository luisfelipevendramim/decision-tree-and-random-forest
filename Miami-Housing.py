# -*- coding: utf-8 -*-
#%% Lendo os dados e armazenando no DataFrame intitulado miami-housing

!pip install pandas

import pandas as pd

#%% Carregando e explorando os dados com os seguintes comandos:

miami_data = pd.read_csv('miami-housing.csv')

miami_data.columns

print(miami_data.head())

miami_data.info()

resumo = miami_data.describe()

#%% Selecionando o alvo da previsão

y = miami_data.SALE_PRC

#%% Escolhendo os Atributos

miami_atributos = ['LND_SQFOOT', 'TOT_LVG_AREA', 'CNTR_DIST', 'age', 'structure_quality']

# Por convenção, esses dados são chamados de X:
X = miami_data[miami_atributos]

X.describe()

#%% Construindo o modeloa de prendizado supervisionado Árvore de Decisão

# Instalando a biblioteca scikit-learn
!pip install scikit-learn

# Importando o modelo 
from sklearn.tree import DecisionTreeRegressor

#Definindo e ajustando o modelo (é especificado um número para random_state para garantir os mesmos resultados em cada execução)
miami_model = DecisionTreeRegressor(random_state=1)
miami_model.fit(X, y)

#%% Validando o modelo

# Existem muitas métricas para resumir a qualidade do modelo, aqui usarei uma chamada Erro Médio Absoluto (também chamada MAE).
from sklearn.metrics import mean_absolute_error

# Calculando o Erro Médio Absoluto
predicted_home_prices = miami_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#%% Validando Dados

# Biblioteca scikit-learn possui uma função train_test_split para dividir os dados em duas partes:
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
miami_model = DecisionTreeRegressor()
miami_model.fit(train_X, train_y)

# Obtendo preços previstos nos dados de validação
val_predictions = miami_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#%% Overfitting e Underfitting

# Criando uma função para ajudar a comparar pontuações MAE de diferentes valores para max_leaf_nodes:
# max_leaf_nodes: fornece uma maneira muito sensata de controlar overfitting vs underfitting
    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Usando um loop for para comparar a precisão de modelos construídos com valores diferentes para max_leaf_nodes:
for max_leaf_nodes in [5, 50, 500, 1000, 2500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#%% Modelo Random Forests

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Definindo, ajustando e validando o modelo
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
miami_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, miami_preds))

#%% Comparando as previsões com os valores reais

# Verificando as previsões comparando com os valores reais
print("Previsões:", miami_preds[:10])  # Exibe as 10 primeiras previsões
print("Valores reais:", val_y[:10])  # Exibe os 10 primeiros valores reais
