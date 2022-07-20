import matplotlib.pyplot as plt
import numpy as np


x = [0, 5, 10, 15, 20, 25, 30, 35,60]
y = [0, 0, 0, 0, 1, 1, 1, 1,1]
fig = plt.figure(figsize=(6.4, 4.8), dpi=150)
plt.axhline(y=0.5, xmin=0, linestyle='--', color='gray')
plt.axvline(x=30, ymin=0.5, linestyle='--', color='gray')
plt.plot(x, y, marker='o')
plt.scatter([20,25], [1,1], s=50, marker='o', color='r', zorder=5)
plt.xlabel("Hours")
plt.ylabel("Result")
plt.xticks(np.arange(0,65,5))
plt.xlim([0,65])
plt.yticks([0, 0.5, 1])

plt.plot([0, 60], [0, 1], label="Wx")
plt.legend()
# plt.show()
plt.savefig('./mlstudy02-fig2.jpg')
