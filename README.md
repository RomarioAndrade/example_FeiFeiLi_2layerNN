# Exampla_FeiFeiLi_2_layer_NN

Implementação, em NumPy puro, de uma rede neural de **2 camadas** treinada com
**gradiente descendente** e **backpropagation** feito à mão. O exemplo é baseado
no código mostrado nas aulas da Fei-Fei Li (Stanford CS231n).

## Como executar

```bash
pip install numpy
python example.py
```

O script imprime, a cada iteração, o número da iteração e o valor da perda
(`loss`), que deve diminuir ao longo do treinamento.

## Explicação do código

### 1. Dimensões e dados

```python
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
```

| Variável | Significado                              | Valor |
|----------|------------------------------------------|-------|
| `N`      | número de exemplos (tamanho do batch)    | 64    |
| `D_in`   | dimensão da entrada                      | 1000  |
| `H`      | tamanho da camada oculta                 | 100   |
| `D_out`  | dimensão da saída                        | 10    |

`x` (64×1000) e `y` (64×10) são dados aleatórios: a rede vai aprender a mapear
`x` em `y`, ou seja, a "decorar" esses dados. O objetivo é didático — mostrar o
mecanismo de treino, não resolver um problema real.

### 2. Pesos

```python
w1, w2 = randn(D_in, H), randn(H, D_out)
```

`w1` (1000×100) liga a entrada à camada oculta de `H` neurônios e `w2`
(100×10) liga a camada oculta à saída. Os pesos começam aleatórios.

### 3. Forward pass (propagação para frente)

```python
h = 1/(1+np.exp(-np.dot(x, w1)))
y_pred = h.dot(w2)
```

- `x · w1`: combinação linear da entrada.
- `1/(1+e^-z)`: função de ativação **sigmoide**, que leva cada valor para o
  intervalo (0, 1) e dá não‑linearidade à rede. O resultado `h` é a saída da
  camada oculta.
- `h · w2`: camada de saída (linear), que produz a previsão `y_pred`.

### 4. Função de perda

```python
loss = np.square(y_pred - y).sum()
```

Soma dos erros quadráticos (*sum of squared errors*) entre a previsão e o valor
real. Quanto menor, melhor a rede se ajusta aos dados.

### 5. Backward pass (backpropagation)

Aplica a regra da cadeia, da saída para a entrada, para obter o gradiente da
perda em relação a cada peso:

```python
grad_y_pred = 2 * (y_pred - y)              # dL/dy_pred
grad_w2 = h.T.dot(grad_y_pred)              # dL/dw2
grad_h = grad_y_pred.dot(w2.T)              # dL/dh
grad_w1 = x.T.dot(grad_h * h * (1 - h))     # dL/dw1
```

- `2 * (y_pred - y)`: derivada de `(y_pred - y)²`.
- `grad_w2`: como `y_pred = h · w2`, o gradiente de `w2` é `hᵀ · grad_y_pred`.
- `grad_h`: o erro é propagado de volta para a camada oculta através de `w2ᵀ`.
- `h * (1 - h)`: derivada da sigmoide (`σ'(z) = σ(z)·(1 − σ(z))`).
- `grad_w1`: gradiente de `w1`, combinando a entrada `x` com o erro que chega
  na camada oculta.

### 6. Atualização dos pesos (gradiente descendente)

```python
w1 -= 1e-4 * grad_w1
w2 -= 1e-4 * grad_w2
```

Cada peso dá um pequeno passo na direção oposta ao gradiente, o que reduz a
perda. `1e-4` é a **taxa de aprendizado** (*learning rate*).

### 7. Loop de treinamento

Os passos 3 a 6 são repetidos **2000 vezes** (`for i in range(2000)`). A cada
iteração a perda tende a diminuir, mostrando que a rede está aprendendo.

## Resumo do fluxo

```
x ──(w1)──> sigmoide ──> h ──(w2)──> y_pred ──> loss
                                                  │
        atualiza w1 <── grad_w1 <── grad_h <── grad_y_pred ──> grad_w2 ──> atualiza w2
```
