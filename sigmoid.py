import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.axvline(0, linestyle='--', color='gray')
plt.show()


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)
plt.plot(x, y1, linestyle='--')
plt.plot(x, y2)  # W의 값이 1일때
plt.plot(x, y3, linestyle='--')
plt.axvline(0, linestyle='--', color='gray')
plt.show()