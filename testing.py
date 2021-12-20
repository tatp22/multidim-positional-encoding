import tensorflow as tf
import torch
import numpy as np
from positional_encodings import TFPositionalEncoding2D
from positional_encodings import PositionalEncoding2D

p_enc_2d = TFPositionalEncoding2D(170)
y = tf.zeros((1,1,1024,170))
assert p_enc_2d(y).shape == (1, 1, 1024, 170)

tf_enc_2d = TFPositionalEncoding2D(123)
pt_enc_2d = PositionalEncoding2D(123)

sample = np.random.randn(2,123,321,170)

tf_out = tf_enc_2d(sample)
pt_out = pt_enc_2d(torch.tensor(sample))

# There is some rounding discrepancy
assert np.sum(np.abs(tf_out.numpy()- pt_out.numpy()) > 0.0001) == 0