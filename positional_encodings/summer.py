import tensorflow as tf
import torch.nn as nn


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        return tensor + penc


class TFSummer(tf.keras.layers.Layer):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(TFSummer, self).__init__()
        self.penc = penc

    @tf.function
    def call(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.shape == penc.shape
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        return tensor + penc
