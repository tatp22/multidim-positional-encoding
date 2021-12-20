import tensorflow as tf
from positional_encodings import TFPositionalEncoding2D

p_enc_2d = TFPositionalEncoding2D(170)
y = tf.zeros((1,1,1024,170))
assert p_enc_2d(y).shape == (1, 1, 1024, 170)