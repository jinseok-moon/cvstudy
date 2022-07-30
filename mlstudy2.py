import torch
import torch.optim as optim

X = torch.Tensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
y = torch.Tensor([[0], [0], [0], [1], [1], [1]])
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)  # Bias 추가
optimizer = optim.SGD([W, b], lr=1)
num_epoch = 1000

for epoch in range(num_epoch+1):
    h = torch.sigmoid(torch.matmul(X, W)+b)  # XW + b
    cost = -torch.mean(y*torch.log(h)+(1-y)*torch.log(1-h))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{num_epoch} Cost: {cost.data}')


print("--- Final Prediction ---")
final_h = torch.sigmoid(torch.matmul(X, W)+b)
prediction = final_h >= 0.5
print(prediction)
