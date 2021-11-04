# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # CNN 02 - Forward/backward propagation
# Last updated: 21.11.02
# Written by Jinseok Moon

# + language="javascript"
# MathJax.Hub.Config({
#     TeX: { equationNumbers: { autoNumber: "AMS" } }
# });
#

# +
import numpy as np

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

# Input layer
x_input = np.array([2, 4])  # x1, x2
y_true = [0.8, 0.4]  # y1. y2: ground truth
x = x_input.reshape(-1)

# First layer
w1 = np.array([0.7, 0.3]) # w11, w21
w2 = np.array([0.2, 0.4]) # w12, w22

W = np.array([w1, w2])
b = np.array([b1, b1])
g1x = np.dot(W, x)
h1x = sigmoid(g1x)

print("g1 =",g1x) # g1 = [2.6 2. ]
print("h1 =", h1x) # h1 = [0.93086158 0.88079708]

# +
# Second layer

w3 = np.array([0.1, 0.5]) # w41, w42
w4 = np.array([0.8, 0.4]) # w51, w52

W_ = np.array([w3, w4])
b_ = np.array([b2, b2])
g2x = np.dot(W_, h1x)
h2x = sigmoid(g2x)
y_predict = h2x  # [y1p, y2p]

print("g2 =",g2x) # g2 = [0.5334847  1.09700809]
print("h2 =", h2x) # h2 = [0.63029549 0.74969909]
# -

# \begin{equation}\label{eqn:eq1}
# \begin{split}
# g_1(x) &=  \sum^{n}_{i} x_iw_i + b = W_1^Tx \\[5pt]
#     &= \left( \begin{matrix}  w_{11} & w_{21} \newline w_{12} & w_{22} \end{matrix} \right) \left( \begin{matrix}  x_{1} \newline x_{2} \end{matrix} \right)
# \end{split}
# \end{equation}
#
# 이번에는 활성화 함수로 계단함수가 아닌 sigmoid function $\sigma(x)$를 이용한다.
#
# \begin{align}
# \label{eqn:eq2} h(x) &= \sigma(x) = {1 \over 1 + e^{-x}} \\[5pt]
# \label{eqn:eq3} \hat{y} &= h(x)
# \end{align}
#
#
#

# 최종 계산해보면, $\hat{y_1}=0.630, \hat{y_2}=0.750$을 얻을 수 있다. 실제 $y_1=0.8, y_2=0.4$라고 가정했을 때, 퍼셉트론에서 예측한 값과 오차가 발생하게 된다.

# ## 비용함수 (Cost function)
# 우리는 퍼셉트론의 순전파/역전파를 이용해서 가중치들을 조정해 줄 것이다. 그리고 이 때 필요한 것이 바로 비용함수와 미분이다.
