import torch
from pos_enc_multidim import PositionalEmbedding1D, PositionalEmbedding2D, PositionalEmbedding3D

p_enc_1d = PositionalEmbedding1D(10)
x = torch.zeros((1,6,10))
print(p_enc_1d(x).shape) # (1, 6, 10)

p_enc_2d = PositionalEmbedding2D(8)
y = torch.zeros((1,6,2,8))
print(p_enc_2d(y).shape) # (1, 6, 2, 8)

p_enc_3d = PositionalEmbedding3D(11)
z = torch.zeros((1,5,6,4,11))
print(p_enc_3d(z).shape) # (1, 5, 6, 4, 11)
