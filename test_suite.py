import numpy as np
import tensorflow as tf
import torch

from positional_encodings import *

tf.config.experimental_run_functions_eagerly(True)


def test_torch_1d_correct_shape():
    p_enc_1d = PositionalEncoding1D(10)
    x = torch.zeros((1, 6, 10))
    assert p_enc_1d(x).shape == (1, 6, 10)

    p_enc_1d = PositionalEncodingPermute1D(10)
    x = torch.zeros((1, 10, 6))
    assert p_enc_1d(x).shape == (1, 10, 6)


def test_torch_2d_correct_shape():
    p_enc_2d = PositionalEncoding2D(170)
    y = torch.zeros((1, 1, 1024, 170))
    assert p_enc_2d(y).shape == (1, 1, 1024, 170)

    p_enc_2d = PositionalEncodingPermute2D(169)
    y = torch.zeros((1, 169, 1, 1024))
    assert p_enc_2d(y).shape == (1, 169, 1, 1024)


def test_torch_3d_correct_shape():
    p_enc_3d = PositionalEncoding3D(125)
    z = torch.zeros((3, 5, 6, 4, 125))
    assert p_enc_3d(z).shape == (3, 5, 6, 4, 125)

    p_enc_3d = PositionalEncodingPermute3D(11)
    z = torch.zeros((7, 11, 5, 6, 4))
    assert p_enc_3d(z).shape == (7, 11, 5, 6, 4)


def test_tf_1d_correct_shape():
    pass  # TODO


def test_tf_2d_correct_shape():
    p_enc_2d = TFPositionalEncoding2D(170)
    y = tf.zeros((1, 1, 1024, 170))
    assert p_enc_2d(y).shape == (1, 1, 1024, 170)


def test_tf_3d_correct_shape():
    pass  # TODO


def test_torch_tf_1d_same():
    pass  # TODO


def test_torch_tf_2d_same():
    tf_enc_2d = TFPositionalEncoding2D(123)
    pt_enc_2d = PositionalEncoding2D(123)

    sample = np.random.randn(2, 123, 321, 170)

    tf_out = tf_enc_2d(sample)
    pt_out = pt_enc_2d(torch.tensor(sample))

    # There is some rounding discrepancy
    assert np.sum(np.abs(tf_out.numpy() - pt_out.numpy()) > 0.0001) == 0


def test_torch_tf_3d_same():
    pass  # TODO


def test_torch_summer():
    model_with_sum = Summer(PositionalEncoding2D(125))
    model_wo_sum = PositionalEncoding2D(125)
    z = torch.rand(3, 5, 6, 125)
    assert (
        np.sum(np.abs((model_wo_sum(z) + z).numpy() - model_with_sum(z).numpy()))
        < 0.0001
    ), "The summer is not working properly!"


def test_tf_summer():
    model_with_sum = TFSummer(TFPositionalEncoding2D(125))
    model_wo_sum = TFPositionalEncoding2D(125)
    z = np.random.randn(3, 5, 6, 125)
    assert (
        np.sum(np.abs((model_wo_sum(z) + z).numpy() - model_with_sum(z).numpy()))
        < 0.0001
    ), "The tf summer is not working properly!"
