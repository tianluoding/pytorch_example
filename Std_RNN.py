import torch.nn as nn
import torch
from torch.autograd import Variable

# 验证标准RNN的输入输出以及隐藏状态的结构
basic_rnn = nn.RNN(input_size=20, hidden_size=50, num_layers=2)
print(basic_rnn.weight_ih_l0.shape)

toy_input = Variable(torch.randn(100, 32, 20))
h_0 = Variable(torch.randn(2, 32, 50))

toy_output, h_n = basic_rnn(toy_input, h_0)
print(toy_output.shape)
print(h_n.shape)

lstm = nn.LSTM(input_size=20, hidden_size=50, num_layers=2)
lstm_out, (h_n, c_n) = lstm(toy_input)
print(lstm_out.shape)
print(h_n.shape)
print(h_0.shape)