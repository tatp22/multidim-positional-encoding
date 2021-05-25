import torch
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D

p_enc_1d = PositionalEncoding1D(10)
x = torch.zeros((1,6,10))
assert p_enc_1d(x).shape == (1,6,10)

p_enc_2d = PositionalEncoding2D(170)
y = torch.zeros((1,1,1024,170))
assert p_enc_2d(y).shape == (1, 1, 1024, 170)

p_enc_3d = PositionalEncoding3D(125)
z = torch.zeros((3,5,6,4,125))
assert p_enc_3d(z).shape == (3, 5, 6, 4, 125)

print("Tests Passed")
