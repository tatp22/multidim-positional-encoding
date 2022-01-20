import numpy as np
import tensorflow as tf
import torch
import time

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
    p_enc_1d = TFPositionalEncoding1D(170)
    x = tf.zeros((1, 1024, 170))
    assert p_enc_1d(x).shape == (1, 1024, 170)


def test_tf_2d_correct_shape():
    p_enc_2d = TFPositionalEncoding2D(170)
    y = tf.zeros((1, 1, 1024, 170))
    assert p_enc_2d(y).shape == (1, 1, 1024, 170)


def test_tf_3d_correct_shape():
    p_enc_3d = TFPositionalEncoding3D(170)
    z = tf.zeros((1, 4, 1, 1024, 170))
    assert p_enc_3d(z).shape == (1, 4, 1, 1024, 170)


def test_torch_tf_1d_same():
    tf_enc_1d = TFPositionalEncoding1D(123)
    pt_enc_1d = PositionalEncoding1D(123)

    sample = np.random.randn(2, 15, 123)

    tf_out = tf_enc_1d(sample)
    pt_out = pt_enc_1d(torch.tensor(sample))

    # There is some rounding discrepancy
    assert np.sum(np.abs(tf_out.numpy() - pt_out.numpy()) > 0.0001) == 0


def test_torch_tf_2d_same():
    tf_enc_2d = TFPositionalEncoding2D(123)
    pt_enc_2d = PositionalEncoding2D(123)

    sample = np.random.randn(2, 123, 321, 170)

    tf_out = tf_enc_2d(sample)
    pt_out = pt_enc_2d(torch.tensor(sample))

    # There is some rounding discrepancy
    assert np.sum(np.abs(tf_out.numpy() - pt_out.numpy()) > 0.0001) == 0


def test_torch_tf_3d_same():
    tf_enc_3d = TFPositionalEncoding3D(123)
    pt_enc_3d = PositionalEncoding3D(123)

    sample = np.random.randn(2, 123, 24, 21, 10)

    tf_out = tf_enc_3d(sample)
    pt_out = pt_enc_3d(torch.tensor(sample))

    # There is some rounding discrepancy
    assert np.sum(np.abs(tf_out.numpy() - pt_out.numpy()) > 0.0001) == 0


def test_torch_summer():
    model_with_sum = Summer(PositionalEncoding2D(125))
    model_wo_sum = PositionalEncoding2D(125)
    z = torch.rand(3, 5, 6, 125)
    assert (
        np.sum(np.abs((model_wo_sum(z) + z).numpy() - model_with_sum(z).numpy()))
        < 0.0001
    ), "The summer is not working properly!"


def test_torch_fixed_1D_encoding():
    embeding_dim = 64
    shape = (13, )
    batch_sizes = (9, 10, 13, 16)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    enc = PositionalEncoding1D(embeding_dim)
    enc.to(device)
    fixed_enc = FixEncoding(enc, shape)
    fixed_enc.to(device)
    
    for batch_size in batch_sizes:
    
        data = torch.randn(batch_size, *shape, embeding_dim).to(device)

        out_fixed = fixed_enc(data)
        out_original = enc(data)

        assert torch.sum(out_original - out_fixed) == 0, "The output of the 1D Positional encoder and the fixed wrapper are not the same. At batch size {batch_size}"

def test_torch_fixed_2D_encoding():
    embeding_dim = 64
    batch_sizes = (9, 10, 13, 16)
    shape = (13, 13)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    enc = PositionalEncoding2D(embeding_dim)
    enc.to(device)
    fixed_enc = FixEncoding(enc, shape)
    fixed_enc.to(device)
    
    for batch_size in batch_sizes:
        data = torch.randn(batch_size, *shape, embeding_dim).to(device)
        out_fixed = fixed_enc(data)
        out_original = enc(data)

        assert torch.sum(out_original - out_fixed) == 0, f"The output of the 2D Positional encoder and the fixed wrapper are not the same. At batch size {batch_size}"

def test_torch_fixed_3D_encoding():
    embeding_dim = 64
    batch_sizes = (9, 10, 13, 16)
    shape = (13, 13, 13)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    enc = PositionalEncoding3D(embeding_dim)
    enc.to(device)
    fixed_enc = FixEncoding(enc, shape)
    fixed_enc.to(device)
    
    for batch_size in batch_sizes:
        data = torch.randn(batch_size, *shape, embeding_dim).to(device)
        out_fixed = fixed_enc(data)
        out_original = enc(data)
        print(out_fixed.shape)
        print(out_original.shape)
        assert torch.sum(out_original - out_fixed) == 0, f"The output of the 2D Positional encoder and the fixed wrapper are not the same. At batch size {batch_size}"

def test_tf_summer():
    model_with_sum = TFSummer(TFPositionalEncoding2D(125))
    model_wo_sum = TFPositionalEncoding2D(125)
    z = np.random.randn(3, 5, 6, 125)
    assert (
        np.sum(np.abs((model_wo_sum(z) + z).numpy() - model_with_sum(z).numpy()))
        < 0.0001
    ), "The tf summer is not working properly!"
