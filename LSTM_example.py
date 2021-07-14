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


'''
data = pd.read_csv("data.csv", usecols=[1]) # usecols参数读取df中的指定列
'''

# 自定义LSTM模型

class lstmModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit(np.sqrt(1.0/hidden_dim))

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        # out = out[12:]
        out = self.fcLayer(out)
        return out

    def weightInit(self, gain):
        for name, param in self.named_parameters():
            if 'lstmLayer.weight' in name:
                init.orthogonal_(param)


lstm = cudAvl(lstmModel(1, 5, 1, 2))

criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=1e-2)

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
    output = lstm(inputs)
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


lstm = lstm.eval()

# 预测结果并比较

px = real[:-1].reshape(-1, 1, 1)
px = torch.from_numpy(px)
ry = real[1:].reshape(-1)
varX = cudAvl(Variable(px, volatile=True))
py = lstm(varX).data
py = py.cpu().numpy()
py = py.reshape(-1)
print(px.shape, py.shape, ry.shape)

# 画出实际结果和预测的结果
plt.plot(py[-24:], 'r', label='prediction')
plt.plot(ry[-24:], 'b', label='real')
plt.legend(loc='best')
plt.show()