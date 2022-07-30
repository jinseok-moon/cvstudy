import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def cost(h, y):
    if y == 1:
        return -np.log(h)
    else:
        return -(1-y)*np.log(1-h)


h = np.linspace(0, 1)
cost_0 = cost(h, 0)
cost_1 = cost(h, 1)

# plt.figure(figsize=(6.4, 6.4), dpi=150)
# plt.plot(h, cost_0, label='-log(1-H(x))')
# plt.plot(h, cost_1, label='-log(H(x))')
# plt.xlabel("H(x)")
# plt.ylabel("Cost")
# plt.legend()
# plt.savefig("sigmoid-cost.jpg")

#
# plt.figure(figsize=(6.4, 4.8), dpi=150)
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)
# plt.plot(x, y1, label='w=0.5', linestyle='--')
# plt.plot(x, y2, label='w=1')  # W의 값이 1일때
# plt.plot(x, y3, label='w=2', linestyle='--')
# plt.axvline(0, linestyle='--', color='gray')
# plt.legend()
# plt.savefig("sigmoid.jpg")


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x-1)
y2 = sigmoid(x)
y3 = sigmoid(x+1)
plt.figure(figsize=(6.4, 4.8), dpi=150)
plt.plot(x, y1,  linestyle='--', label='b=-1')  # x + 0.5
plt.plot(x, y2, label='b=0')  # x + 1
plt.plot(x, y3, linestyle='--', label='b=1') # x + 1.5
plt.axvline(0, linestyle='--', color='gray')
plt.title('Sigmoid Function')
plt.legend()
plt.savefig('sigmoid-bias.jpg')