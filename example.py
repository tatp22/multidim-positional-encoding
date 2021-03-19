import torch
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D

p_enc_1d = PositionalEncoding1D(10)
x = torch.zeros((1,6,10))
assert p_enc_1d(x).shape == (1,6,10)

p_enc_2d = PositionalEncoding2D(8)
y = torch.zeros((1,6,2,8))
assert p_enc_2d(y).shape == (1, 6, 2, 8)

p_enc_3d = PositionalEncoding3D(11)
z = torch.zeros((3,5,6,4,11))
assert p_enc_3d(z).shape == (3, 5, 6, 4, 11)

print("Tests Passed")
