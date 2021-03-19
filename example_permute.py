import torch
from positional_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D, PositionalEncodingPermute3D

p_enc_1d = PositionalEncodingPermute1D(10)
x = torch.zeros((1,10,6))
assert p_enc_1d(x).shape ==(1, 10, 6)

p_enc_2d = PositionalEncodingPermute2D(8)
y = torch.zeros((1,8,6,2))
assert p_enc_2d(y).shape == (1, 8, 6, 2)

p_enc_3d = PositionalEncodingPermute3D(11)
z = torch.zeros((7,11,5,6,4))
assert p_enc_3d(z).shape == (7, 11, 5, 6, 4)

print("Tests Passed")
