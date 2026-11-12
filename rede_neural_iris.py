"""
Rede Neural Artificial (1 camada escondida) implementada do zero com NumPy
para classificar as espécies do conjunto de dados Iris.

- NumPy: todas as operações matriciais (forward, backward, atualização).
- Scikit-Learn: usado APENAS para carregar e preparar os dados.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SEED = 42
np.random.seed(SEED)

iris = load_iris()
X = iris.data                      # (150, 4)
y = iris.target.reshape(-1, 1)     # (150, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder(sparse_output=False)
Y_train = encoder.fit_transform(y_train)   # (120, 3)
Y_test = encoder.transform(y_test)         # (30, 3)

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

n_entrada = X_train.shape[1]   # 4
n_escondida = 8
n_saida = Y_train.shape[1]     # 3

W1 = np.random.randn(n_entrada, n_escondida) * np.sqrt(2.0 / n_entrada)
b1 = np.zeros((1, n_escondida))
W2 = np.random.randn(n_escondida, n_saida) * np.sqrt(2.0 / n_escondida)
b2 = np.zeros((1, n_saida))


def relu(z):
    return np.maximum(0, z)


def relu_derivada(z):
    return (z > 0).astype(float)


def softmax(z):
    z_deslocado = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_deslocado)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1          # (n, 8)
    A1 = relu(Z1)             # (n, 8)
    Z2 = A1 @ W2 + b2         # (n, 3)
    A2 = softmax(Z2)          # (n, 3) probabilidades
    return Z1, A1, Z2, A2


def cross_entropy(Y, A2):
    eps = 1e-12  
    return -np.mean(np.sum(Y * np.log(A2 + eps), axis=1))


def acuracia(Y, A2):
    return np.mean(np.argmax(A2, axis=1) == np.argmax(Y, axis=1))

epocas = 1000
taxa_aprendizado = 0.1
m = X_train.shape[0]

for epoca in range(1, epocas + 1):
    Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

    perda = cross_entropy(Y_train, A2)

    dZ2 = (A2 - Y_train) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivada(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 -= taxa_aprendizado * dW1
    b1 -= taxa_aprendizado * db1
    W2 -= taxa_aprendizado * dW2
    b2 -= taxa_aprendizado * db2

    if epoca == 1 or epoca % 100 == 0:
        print(f"Época {epoca:4d}/{epocas} | Perda: {perda:.4f} | "
              f"Acurácia treino: {acuracia(Y_train, A2) * 100:.2f}%")

_, _, _, A2_teste = forward(X_test, W1, b1, W2, b2)
perda_teste = cross_entropy(Y_test, A2_teste)
acc_teste = acuracia(Y_test, A2_teste)

print("\n=== Avaliação final ===")
print(f"Perda no teste: {perda_teste:.4f}")
print(f"Acurácia no teste: {acc_teste * 100:.2f}%")
