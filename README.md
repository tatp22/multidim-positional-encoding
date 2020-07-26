# 3D Sinusodal Pytorch Positional encoding (WIP)

This is an implemenation of 3d sinusodal positional encoding, being able to encode on tensors of the form `(batchsize, x, y, z, ch)`, where the positional embeddings will be added to the `ch` dimension. The [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) allowed for positional encoding in only one dimension, however, this works to extend this to three dimensions.

Specifically, the formula for inserting the positional encoding will be as follows:

TODO

Don't worry if the input is not divisible by the model dim; all necessary padding will be taken care of.

## Usage:

```
import torch

tensor = torch.zeros((4,8,16,32,64)) # Or whatever size you want it to be
positional_encoding = pos_enc_3d(tensor)
```

## Thank you

Thank you for [this](https://github.com/wzlxjtu/PositionalEncoding2D) repo for inspriration of this method.
