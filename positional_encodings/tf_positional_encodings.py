import numpy as np
import tensorflow as tf


class TFPositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.

        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".

        """
        super(TFPositionalEncoding1D, self).__init__()

        self.channels = int(np.ceil(channels / 2) * 2)
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = tf.expand_dims(tf.concat((tf.sin(sin_inp_x), tf.cos(sin_inp_x)), -1), 0)
        emb = emb[0]  # A bit of a hack
        return tf.repeat(emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0)


class TFPositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.

        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".

        """
        super(TFPositionalEncoding2D, self).__init__()

        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        _, x, y, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(tf.concat((tf.sin(sin_inp_x), tf.cos(sin_inp_x)), -1), 1)
        emb_y = tf.expand_dims(tf.concat((tf.sin(sin_inp_y), tf.cos(sin_inp_y)), -1), 0)
        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), -1)
        return tf.repeat(emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0)


class TFPositionalEncoding3D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.

        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".

        """
        super(TFPositionalEncoding3D, self).__init__()

        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(inputs.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        _, x, y, z, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)
        pos_z = tf.range(z, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = tf.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = tf.expand_dims(
            tf.expand_dims(tf.concat((tf.sin(sin_inp_x), tf.cos(sin_inp_x)), -1), 1), 1
        )
        emb_y = tf.expand_dims(
            tf.expand_dims(tf.concat((tf.sin(sin_inp_y), tf.cos(sin_inp_y)), -1), 1), 0
        )
        emb_z = tf.expand_dims(
            tf.expand_dims(tf.concat((tf.sin(sin_inp_z), tf.cos(sin_inp_z)), -1), 0), 0
        )

        emb_x = tf.tile(emb_x, (1, y, z, 1))
        emb_y = tf.tile(emb_y, (x, 1, z, 1))
        emb_z = tf.tile(emb_z, (x, y, 1, 1))

        emb = tf.concat((emb_x, emb_y, emb_z), -1)
        return tf.repeat(emb[None, :, :, :, :org_channels], tf.shape(inputs)[0], axis=0)
