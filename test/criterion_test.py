import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
a = torch.tensor([[0.2,0.3,0.5]]).float()
target = torch.tensor([0]).long()

for i in range(target.shape[0]):
    ta = target[i].unsqueeze(0)
    loss = criterion(a, ta)
    print(loss)
