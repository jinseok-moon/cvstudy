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

def sigmoid(x, diff=False):
    if diff==True:
        return sigmoid(x)*(1-sigmoid(x));
    else:
        return 1 / (1 +np.exp(-x))

# Input layer
x_input = np.array([2, 4])  # x1, x2
y_true = [0.8, 0.4]  # y1. y2: ground truth
x = x_input.reshape(-1)

# First layer
w1 = np.array([0.7, 0.3]) # w11, w21
w2 = np.array([0.2, 0.4]) # w12, w22

W = np.array([w1, w2])
g = np.dot(W, x)
h = sigmoid(g)

print("g =",g) # g = [2.6 2. ]
print("h =", h) # h = [0.93086158 0.88079708]

# +
# Second layer

w3 = np.array([0.1, 0.5]) # w41, w42
w4 = np.array([0.8, 0.4]) # w51, w52

W_2 = np.array([w3, w4])
z = np.dot(W_2, h)
y = sigmoid(z)
y_predict = y  # [y1p, y2p]

print("z =", z) # g2 = [0.5334847  1.09700809]
print("y =", y) # h2 = [0.63029549 0.74969909]


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

# 최종 계산해보면, $\hat{y_1}=0.630, \hat{y_2}=0.750$을 얻을 수 있다. 실제 $y_1=0.8, y_2=0.4$라고 가정했을 때, 퍼셉트론에서 예측한 값과 오차가 발생하게 된다.

# ## 비용함수 (Cost function)
# 우리는 퍼셉트론의 순전파/역전파를 이용해서 가중치들을 조정해 줄 것이다. 그리고 이 때 필요한 것이 바로 비용함수와 미분이다.
# $\sigma(x)$의 미분은 $\sigma(x)(1-\sigma(x)$)로 나타낼 수 있다. 유도식은 추후 추가.
# 비용함수는 C는 아래와 같이 정의한다. 
# \begin{align}
# \label{eqn:eq2} C &= {1 \over 2} \left[ (\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2\right]
# \end{align}

def cost(y_true, y_predict, length):
    cost = 0
    for n in range(length):
        cost += 1/2*(y_true[n]-y_predict[n])**2
    return cost


cost(y_predict, y_true, len(y_true))

# \begin{equation}\label{eqn:eq1}
# \begin{split}
# z &= W_2^T h, & {dz \over dW_2}&= h\\[5pt]
# \left( \begin{matrix}  z_{1} \newline z_{2} \end{matrix} \right)  &=  \left( \begin{matrix}  w_{31} & w_{41} \newline w_{32} & w_{42} \end{matrix} \right) \left( \begin{matrix}  h_{1} \newline h_{2} \end{matrix} \right) \\[5pt]
# z_1 &= w_{31}h_1 + w_{41}h_2 \\[5pt]
# z_2 &= w_{32}h_1 + w_{42}h_2 \\[5pt]
# {dz_1 \over dw_{31}}&= h_1, \enspace {dz_1 \over dw_{41}}= h_2 \\[5pt]
# {dz_2 \over dw_{32}}&= h_1, \enspace {dz_1 \over dw_{42}}= h_2 \\[5pt]
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

# ## References
# [<sup id="fn1">1</sup>](#fn1-back) 선형대수와 통계학으로 배우는 머신러닝 with Python. https://github.com/bjpublic/MachineLearning
