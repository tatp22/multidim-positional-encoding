import torch
from pos_enc_3d import PositionalEmbedding3D

p_enc = PositionalEmbedding3D(12)
x = torch.zeros((1,5,6,4,12))
print(p_enc(x)[0,0,2,1])
