import tensorflow as tf
import numpy as np

class TFPositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels:int, return_format:str="pos", dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.

        Keyword Args:
            return_format str: Return either the position encoding "pos" or the sum
                of the inputs with the position encoding "sum". Default is "pos".
            dtype: output type of the encodings. Default is "tf.float32".

        """
        super(TFPositionalEncoding2D, self).__init__()
        if return_format not in ["pos", "sum"]:
            raise ValueError(f'"{return_format}" is an unkown return format. Value must be "pos" or "sum')
        self.return_format = return_format
            
        self.channels = int(2 * np.ceil(channels/4))
        self.inv_freq = np.float32(1 / np.power(10000, np.arange(0, self.channels, 2) / np.float32(self.channels)))
        
    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape)!=4:
            raise RuntimeError("The input tensor has to be 4d!")
        _, x, y, org_channels = inputs.shape
        
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
        pos_enc = tf.repeat(emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0)
        if self.return_format == "pos":
            return pos_enc
        elif self.return_format == "sum":
            return inputs + pos_enc