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
# Basic import
import numpy as np

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

# Input layer
input_data = np.array([2, 4, 3])
x = input_data.reshape(-1)

# First layer
w1 = np.array([0.7, 0.3, 0.9]) # w11, w21, w31
w2 = np.array([0.2, 0.4, 0.1]) # w12, w22, w32
b1 = -0.2

W = np.array([w1, w2])
b = np.array([b1, b1])
g1x = np.dot(W, x) + b
h1x = sigmoid(g1x)

print("g1 =",g1x) # g1 = [3.9 3.6]
print("h1 =", h1x) # h1 = [0.98015969 0.97340301]

# +
# Second layer

w3 = np.array([0.1, 0.5]) # w41, w42
w4 = np.array([0.8, 0.4]) # w51, w52
b2 = 0.5

W_ = np.array([w3, w4])
b_ = np.array([b2, b2])
g2x = np.dot(W_, h1x) + b_
h2x = sigmoid(g2x)

print("g2 =",g2x) # g2 = [1.04484561 1.65151343]
print("h2 =", h2x) # h2 = [0.73978389 0.83909549]
