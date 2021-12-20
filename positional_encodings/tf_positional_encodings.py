
import tensorflow as tf
import numpy as np

class TFPositionalEncoding2D(tf.keras.layers.Layer):

    def __init__(self, channels, dtype=tf.float32):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(TFPositionalEncoding2D, self).__init__()
        self.channels = int(2 * np.ceil(channels/4))
        self.inv_freq = np.float32(1 / np.power(10000, np.arange(0, self.channels, 2) / np.float32(self.channels)))

    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape)!=4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, org_channels = inputs.shape
        dtype = self.inv_freq.dtype
        
        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)
        
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)
        
        emb_x = tf.expand_dims(tf.concat((tf.sin(sin_inp_x), tf.cos(sin_inp_x)), -1),1)
        emb_y = tf.expand_dims(tf.concat((tf.sin(sin_inp_y), tf.cos(sin_inp_y)), -1),0)

        emb_x = tf.tile(emb_x, (1,y,1))
        emb_y = tf.tile(emb_y, (x,1,1))
        emb = tf.concat((emb_x, emb_y),-1)

        return tf.repeat(emb[None, :, :, :org_channels], batch_size, axis=0)
