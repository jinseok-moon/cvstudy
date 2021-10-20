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

# # CNN 01 - Perceptron
# Last updated: 21.10.06
# Written by Jinseok Moon

# + language="javascript"
# MathJax.Hub.Config({
#     TeX: { equationNumbers: { autoNumber: "AMS" } }
# });
#
# -

# ## What is perceptron?
#
# 퍼셉트론[<sup id="fn1-back">1</sup>](#fn1)이란 인공신경망에서 사용하는 최소 단위라고 보면 편하다. 입력값에 대하여 가중합, 활성화 함수를 통해 출력값을 얻어내는 모델이다.  
# Eq.\ref{eqn:eq1}은 가중합을 나타내며, Eq.\ref{eqn:eq2}는 활성화 함수를 나타내며, 퍼셉트론에서는 활성화함수로 계단함수를 이용한다.
#
# \begin{align}\label{eqn:eq1} g(x) &= \sum^{n}_{i} x_iw_i + b \\[5pt]
# \label{eqn:eq2} h(x) &= \begin{cases} 0, & g(x) \leq 0 \newline 1, & g(x) > 0 \end{cases} \\[5pt]
# \label{eqn:eq3} \hat{y} &= h(x)
# \end{align}
#
# <center>
# 	<figure> <img src="./Images/Study/cnn01_01.png" alt="Perceptron" id="fig1"/>
#         <figcaption>Fig.1 퍼셉트론</figcaption>
#     </figure>
# </center>
#
# Fig.[1](#fig1)의 왼쪽 부분처럼 g(x)가 한개뿐인 경우, 단층 퍼셉트론 (Single layer perceptron) 이라고 하며, 여러개를 조합한 경우 다층 퍼셉트론 (Multilayer perceptron, MLP) 이라고 한다.
# 여기서 $b$는 편향치를 의미하며, 편향치에 따라 식(Eq.\ref{eqn:eq2})이 아래와 같이 바뀐다.
#
# \begin{equation}\label{eqn:eq4} h(x) = \begin{cases} 0, & g(x) \leq b \newline 1, & g(x) > b \end{cases} \end{equation}
#
# 위에 나열했던 수식들을 행렬식으로 바꾸어 표현하는것이 일반적이다(Eq.\ref{eqn:eq5}, \ref{eqn:eq6}).
#
# \begin{equation}\label{eqn:eq5} W = \left( \begin{matrix}  w_1 & w_2 \end{matrix} \right) = \left( \begin{matrix}  w_{11} & w_{12} \newline w_{21} & w_{22} \end{matrix} \right), \enspace
# x = \left( \begin{matrix}  x_{1} \newline x_{2} \end{matrix} \right), \enspace
# b = \left( \begin{matrix}  b_{1} \newline b_{2} \end{matrix} \right) \end{equation}
#
# \begin{equation}\label{eqn:eq6}
# \begin{split}
# g &= W^Tx + b \\\\[5pt]
#     &= \left( \begin{matrix}  w_{11} & w_{21} \newline w_{12} & w_{22} \end{matrix} \right) \left( \begin{matrix}  x_{1} \newline x_{2} \end{matrix} \right) + \left( \begin{matrix}  b_{1} \newline b_{2} \end{matrix} \right)
# \end{split}
# \end{equation}
#
# ## Example
# <center>
# 	<figure> <img src="./Images/Study/cnn01_02.png" alt="Perceptron" id="fig2"/>
#         <figcaption>Fig.2 다층 퍼셉트론 예제</figcaption>
#     </figure>
# </center>
#
# 위 Fig.[2](#fig2)와 같은 다층 퍼셉트론의 예를 살펴보자. 퍼셉트론을 행렬식으로 표현하면 아래와 같다.
#
# \begin{equation}\label{eqn:eq7}
# \begin{split}
# g &= W^Tx + b \\\\[5pt]
#     &= \left( \begin{matrix}  0.7 & 0.3 \newline 0.2 & 0.4 \end{matrix} \right) \left( \begin{matrix}  2 \newline 4 \end{matrix} \right) + \left( \begin{matrix} -0.3 \newline 0.1 \end{matrix} \right)
# \end{split}
# \end{equation}
# 계산해보면, $g_1=2.3, g_2=2.1$로 계산되어서, $\hat{y_1}=1, \hat{y_2}=1$을 얻을 수 있다.  
# 아래 코드는 python으로 구현한 것인데, $W$의 행렬이 numpy에서 애초에 계산이 편하게끔 들어가 있기 때문에 그대로 `gx = np.dot(W, x) + b`을 사용했다.

# +
# Basic import
import numpy as np

# Input layer
input_data = np.array([2,4])
x = input_data.reshape(-1)

# Weight & bias
w1 = np.array([0.7, 0.3]) # w11, w21
w2 = np.array([0.2, 0.4]) # w12, w22
b1 = -0.3
b2 = 0.1

# Calculate g(x)
W = np.array([w1, w2])
b = np.array([b1, b2])
gx = np.dot(W, x) + b

# h(x), output layer
hx = gx >= 0 # Step function

print("g =",gx) # [2.3 2.1]
print("h =", hx) # [True, True] [1, 1]
# -

# ## References
# [<sup id="fn1">1</sup>](#fn1-back) Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386–408. https://doi.org/10.1037/h0042519  