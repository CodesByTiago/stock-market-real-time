# Stock Market Real Time
Cria modelo que prevê entrada no mercado de ações brasileiro em tempo real com Python, Machine Learning e TensorFlow.


Para criar um modelo que prevê a entrada no mercado de ações brasileiro em tempo real utilizando Python, Machine Learning e TensorFlow, precisamos considerar o seguinte:

1. **Coleta de dados em tempo real**: Utilizar APIs que fornecem dados em tempo real, como a Alpha Vantage, IEX Cloud ou Yahoo Finance.
2. **Pré-processamento dos dados**: Limpeza e preparação dos dados em tempo real.
3. **Treinamento do modelo**: Treinar o modelo de machine learning.
4. **Predição em tempo real**: Usar o modelo para fazer previsões em tempo real.

### Passo 1: Coleta de Dados em Tempo Real

Usaremos a API do Yahoo Finance para coletar dados em tempo real. Para obter dados em tempo real, você pode usar a biblioteca `yfinance`.

### Passo 2: Pré-processamento dos Dados

Os dados serão pré-processados para serem alimentados no modelo.

### Passo 3: Treinamento do Modelo

Treinaremos o modelo de machine learning com TensorFlow.

### Passo 4: Predição em Tempo Real

Vamos usar o modelo para fazer previsões em tempo real.

Aqui está um exemplo completo:

#### Instalação das Bibliotecas Necessárias

```python
!pip install yfinance pandas tensorflow scikit-learn
```

#### Coleta de Dados Históricos

Primeiro, coletamos dados históricos para treinar o modelo.

```python
import yfinance as yf
import pandas as pd

# Coletar dados históricos
ticker = 'PETR4.SA'
dados = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Adicionar médias móveis e mudança percentual
dados['MA20'] = dados['Close'].rolling(window=20).mean()
dados['MA50'] = dados['Close'].rolling(window=50).mean()
dados['Daily Return'] = dados['Close'].pct_change()

# Remover valores nulos
dados = dados.dropna()

print(dados.head())
```

#### Pré-processamento dos Dados

```python
from sklearn.preprocessing import StandardScaler

# Seleção de recursos
features = ['Close', 'Volume', 'MA20', 'MA50', 'Daily Return']
target = 'Close'

X = dados[features]
y = dados[target]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

#### Treinamento do Modelo

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definir a arquitetura da rede neural
modelo = Sequential()
modelo.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1))

# Compilar o modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
historia = modelo.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

#### Avaliação do Modelo

```python
# Fazer previsões
predicoes = modelo.predict(X_test)

# Avaliar o modelo
mae = tf.keras.metrics.mean_absolute_error(y_test, predicoes).numpy()
mse = tf.keras.metrics.mean_squared_error(y_test, predicoes).numpy()
rmse = mse ** 0.5

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
```

#### Predição em Tempo Real

Para coletar dados em tempo real e fazer previsões em tempo real, podemos configurar um loop que coleta novos dados e faz previsões continuamente.

```python
import time

# Função para obter dados em tempo real
def obter_dados_reais(ticker):
    dados_reais = yf.download(ticker, period='1d', interval='1m')
    dados_reais['MA20'] = dados_reais['Close'].rolling(window=20).mean()
    dados_reais['MA50'] = dados_reais['Close'].rolling(window=50).mean()
    dados_reais['Daily Return'] = dados_reais['Close'].pct_change()
    dados_reais = dados_reais.dropna()
    return dados_reais

# Loop para fazer previsões em tempo real
while True:
    dados_reais = obter_dados_reais(ticker)
    X_reais = dados_reais[features]
    X_reais_scaled = scaler.transform(X_reais)
    
    # Fazer previsão
    predicao_reais = modelo.predict(X_reais_scaled)
    print(f'Predição do preço de fechamento: {predicao_reais[-1][0]}')
    
    # Esperar um minuto antes de coletar novos dados
    time.sleep(60)
```

Este exemplo coleta dados a cada minuto e faz previsões com base nos dados mais recentes. Você pode ajustar o intervalo de tempo e melhorar a robustez do modelo com técnicas adicionais conforme necessário.
