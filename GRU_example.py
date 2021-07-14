import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from seqInit import toTs, cudAvl
from seqInit import input_size
from seqInit import train, real


class GRUModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.GRULayer = nn.GRU(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.GRULayer(x)
        out = self.relu(out)
       # out = out[12:]
        out = self.fcLayer(out)
        return out

gru = cudAvl(GRUModel(1, 5, 1, 2))

criterion = nn.MSELoss()
optimizer = optim.Adam(gru.parameters(), lr=1e-2)

# 处理输入

train = train.reshape(-1, 1, 1)
print(train.shape)
x = torch.from_numpy(train[:-1])
y = torch.from_numpy(train[1:])

# 训练模型

frq, sec = 3500, 350
loss_set = []

for e in range(1, frq + 1) :
    inputs = cudAvl(Variable(x))
    target = cudAvl(Variable(y))
    #forward
    output = gru(inputs)
    loss = criterion(output, target)
    # update paramters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print training information
    print_loss = loss.data
    loss_set.append((e, print_loss))
    if e % sec == 0 :
        print('Epoch[{}/{}], Loss: {:.5f}'.format(e, frq, print_loss))

# 作出损失函数变化图像
pltX = np.array([loss[0] for loss in loss_set])
pltY = np.array([loss[1] for loss in loss_set])
plt.title('loss function output curve')
plt.plot(pltX, pltY)
plt.show()

# 预测结果并比较

px = real[:].reshape(-1, 1, 1)
px = torch.from_numpy(px)
ry = real[:].reshape(-1)
varX = cudAvl(Variable(px))
py = gru(varX).data
py = py.cpu().numpy()
py = py.reshape(-1)
print(px.shape, py.shape, ry.shape)

# 画出实际结果和预测的结果
plt.plot(py, 'r', label='prediction')
plt.plot(ry, 'b', label='real')
plt.legend(loc='best')
plt.show()