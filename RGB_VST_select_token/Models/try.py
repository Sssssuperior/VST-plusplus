import torch.nn as nn
import torch

a = torch.randn(3,1,786,786)
b = torch.randn(3,1,1,784)
a = torch.add(b,a[:,:,:,1:785])
print(a.size())