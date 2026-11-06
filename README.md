# Codigo exemplo De Fei Fei Li

O código reproduz o slide 34 da *Lecture 4: Neural Networks and
Backpropagation* do curso CS231n de Stanford (Fei-Fei Li, Yunzhu Li e Ruohan
Gao, 2023), que mostra que treinar uma rede de 2 camadas cabe em cerca de 20
linhas.

## Como executar

```bash
pip install numpy
python example.py
```


## Explicação do código

O slide 34 divide o código em quatro partes: definir a rede, forward pass,
calcular os gradientes analíticos e gradiente descendente.

### 1. Definir a rede

```python
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)
```

| Variável | Significado                              | Valor |
|----------|------------------------------------------|-------|
| `N`      | número de exemplos (tamanho do batch)    | 64    |
| `D_in`   | dimensão da entrada                      | 1000  |
| `H`      | tamanho da camada oculta                 | 100   |
| `D_out`  | dimensão da saída                        | 10    |


### 2. Forward pass

```python
h = 1/(1+np.exp(-np.dot(x, w1)))
y_pred = h.dot(w2)
loss = np.square(y_pred - y).sum()
```

- `x · w1`: combinação linear da entrada (64×100).
- `1/(1+e^-z)`: sigmoide, que leva cada valor para o intervalo (0, 1). O
  resultado `h` é a saída da camada oculta.
- `h · w2`: camada de saída (linear), que produz a previsão `y_pred` (64×10).
- `loss`: soma dos erros quadráticos entre a previsão e o valor real. Quanto
  menor, melhor a rede se ajusta aos dados.

### 3. Gradientes analíticos (backpropagation)

Percorre o grafo da perda até os pesos, aplicando a regra da cadeia:

```python
grad_y_pred = 2 * (y_pred - y)              # ∂L/∂y_pred
grad_w2 = h.T.dot(grad_y_pred)              # ∂L/∂w2
grad_h = grad_y_pred.dot(w2.T)              # ∂L/∂h
grad_w1 = x.T.dot(grad_h * h * (1 - h))     # ∂L/∂w1
```

- `2 * (y_pred - y)`: derivada de `(y_pred − y)²`.
- `grad_w2` e `grad_h`: regras da multiplicação de matrizes (slide 144) para
  `y_pred = h · w2`, ou seja, `hᵀ · ∂L/∂y_pred` e `∂L/∂y_pred · w2ᵀ`.
- `grad_h * h * (1 - h)`: gradiente de cima × gradiente local da sigmoide
  (slide 94). Leva o erro de `h` para antes da ativação.
- `grad_w1`: de novo a regra da multiplicação de matrizes, agora para `x · w1`.


### 4. Gradiente descendente

```python
w1 -= 1e-4 * grad_w1
w2 -= 1e-4 * grad_w2
```

Cada peso dá um pequeno passo na direção oposta ao gradiente, o que reduz a
perda. `1e-4` é chamado de taxa de aprendizado (*learning rate*).

### Loop de treinamento

Os passos 2 a 4 são repetidos **2000 vezes** (`for i in range(2000)`). A cada
iteração a perda tende a diminuir, mostrando que a rede está aprendendo.


