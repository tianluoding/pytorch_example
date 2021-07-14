import torch
from torch.nn import Embedding
from torch.autograd import Variable

word_to_ix = {'hello': 0, 'world': 1}
embeds = Embedding(2, 5)
hello_ix = torch.LongTensor([word_to_ix['hello']])
hello_ix = Variable(hello_ix)
hello_embed = embeds(hello_ix)
print(hello_embed)