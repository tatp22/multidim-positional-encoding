# 3D Sinusodal Pytorch Positional encoding (WIP)

This is an implemenation of 3d sinusodal positional encoding, being able to encode on tensors of the form `(batchsize, x, y, z, ch)`, where the positional embeddings will be added to the `ch` dimension. The [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) allowed for positional encoding in only one dimension, however, this works to extend this to three dimensions.

Specifically, the formula for inserting the positional encoding will be as follows:

```
PE(x,y,z,2i) = sin(x/10000^(6i/D))
PE(x,y,z,2i+1) = cos(x/10000^(6i/D))
PE(x,y,z,2j+D/3) = sin(y/10000^(6j/D))
PE(x,y,z,2j+1+D/3) = cos(y/10000^(6j/D))
PE(x,y,z,2k+2D/3) = sin(z/10000^(6k/D))
PE(x,y,z,2k+1+2D/3) = cos(z/10000^(6k/D))

Where:
(x,y,z) is a point in 3d space
i,j,k is in [0, D/6), where D is the size of the ch dimension
```

This is just a natural extension of the 3D positional encoding used in [this](https://arxiv.org/pdf/1908.11415.pdf) paper.

Don't worry if the input is not divisible by 6; all necessary padding will be taken care of.

## Usage:

```
import torch

tensor = torch.zeros((4,8,16,32,64)) # Or whatever size you want it to be
positional_encoding = pos_enc_3d(tensor)
```

## Thank you

Thank you for [this](https://github.com/wzlxjtu/PositionalEncoding2D) repo for inspriration of this method.
