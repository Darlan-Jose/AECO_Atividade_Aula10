# Rede Neural do Zero para Classificação do Iris

Implementação, do zero, de uma Rede Neural Artificial com uma camada escondida para classificar as três espécies de flores do conjunto de dados **Iris** (*setosa*, *versicolor* e *virginica*).

- **NumPy** realiza todas as operações da rede (forward pass, backpropagation e atualização dos pesos).
- **Scikit-Learn** é usado estritamente para carregar e preparar os dados.

## Arquitetura

| Camada   | Neurônios | Ativação |
|----------|-----------|----------|
| Entrada  | 4         | (4 características do Iris) |
| Escondida| 8         | ReLU     |
| Saída    | 3         | Softmax  |

## Etapas do script

1. **Pré-processamento:** carrega o Iris com `load_iris`, divide em treino (80%) e teste (20%), padroniza as características (média 0 e desvio padrão 1) e converte os rótulos para One-Hot Encoding.
2. **Inicialização:** pesos (`W1`, `W2`) aleatórios com escala He e vieses (`b1`, `b2`) zerados.
3. **Treinamento:** 1000 épocas de Gradiente Descendente, com forward pass, perda Cross-Entropy, backpropagation pela regra da cadeia e atualização dos parâmetros (taxa de aprendizado de 0,1).
4. **Avaliação:** inferência no conjunto de teste e exibição da perda e da acurácia finais.

## Requisitos

- Python 3.8 ou superior
- `numpy`
- `scikit-learn` (versão 1.2 ou superior, pois o script usa `OneHotEncoder(sparse_output=False)`)

## Como executar

1. Clone o repositório e entre na pasta do projeto.

2. (Opcional) Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute o script:

   ```bash
   python rede_neural_iris.py
   ```

## Exemplo de saída

```
Treino: 120 amostras | Teste: 30 amostras
Época    1/1000 | Perda: 1.5358 | Acurácia treino: 6.67%
...
Época 1000/1000 | Perda: 0.0488 | Acurácia treino: 98.33%

=== Avaliação final ===
Perda no teste: 0.0801
Acurácia no teste: 96.67%
```

## Personalização

Os principais hiperparâmetros ficam no próprio script e podem ser alterados:

- `epocas`: número de épocas de treinamento (padrão: 1000)
- `taxa_aprendizado`: passo do Gradiente Descendente (padrão: 0,1)
- `n_escondida`: número de neurônios da camada escondida (padrão: 8)
- `SEED`: semente aleatória, que garante resultados reproduzíveis (padrão: 42)

> Como o conjunto de teste tem apenas 30 amostras, a acurácia pode variar de forma perceptível ao mudar a semente ou a divisão dos dados.
