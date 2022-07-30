import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)

y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # in-place operation

# a - low level cost assumption
cost_a = (y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
print(cost_a)

# b - F.softmax() + torch.log() = F.log_softmax()
cost_b = (y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
print(cost_b)

# c - F.log_softmax() + F.nll_loss() = F.cross_entropy()
cost_c = F.nll_loss(F.log_softmax(z, dim=1), y)
print(cost_c)

# d - F.cross_entropy()
cost_d = F.cross_entropy(z, y)
print(cost_d)
