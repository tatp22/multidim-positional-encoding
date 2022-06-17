# 1D, 2D, and 3D Sinusoidal Postional Encoding (Pytorch and Tensorflow)

![Code Coverage](./svgs/cov.svg)
[![PyPI version](https://badge.fury.io/py/positional-encodings.svg)](https://badge.fury.io/py/positional-encodings)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a practical, easy to download implemenation of 1D, 2D, and 3D
sinusodial positional encodings for PyTorch and Tensorflow.

It is able to encode on tensors of the form `(batchsize, x, ch)`, `(batchsize,
x, y, ch)`, and `(batchsize, x, y, z, ch)`, where the positional encodings will
be calculated along the `ch` dimension. The [Attention is All You
Need](https://arxiv.org/pdf/1706.03762.pdf) allowed for positional encoding in
only one dimension, however, this works to extend this to 2 and 3 dimensions.

This also works on tensors of the form `(batchsize, ch, x)`, etc. See the usage for more information.

*NOTE*: The import syntax has changed as of version `6.0.1`. See the section for details.

To install, simply run:

```
pip install positional-encodings[pytorch,tensorflow]
```

You can also install the pytorch and tf encodings individually with the following
commands.

* For a PyTorch only installation, run `pip install positional-encodings[pytorch]`
* For a TensorFlow only installation, run `pip install positional-encodings[tensorflow]`

## Usage (PyTorch):

The repo comes with the three main positional encoding models,
`PositionalEncoding{1,2,3}D`. In addition, there are a `Summer` class that adds
the input tensor to the positional encodings.

```python3
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer

# Returns the position encoding only
p_enc_1d_model = PositionalEncoding1D(10)

# Return the inputs with the position encoding added
p_enc_1d_model_sum = Summer(PositionalEncoding1D(10))

x = torch.rand(1,6,10)
penc_no_sum = p_enc_1d_model(x) # penc_no_sum.shape == (1, 6, 10)
penc_sum = p_enc_1d_model_sum(x)
print(penc_no_sum + x == penc_sum) # True
```

```python3
p_enc_2d = PositionalEncoding2D(8)
y = torch.zeros((1,6,2,8))
print(p_enc_2d(y).shape) # (1, 6, 2, 8)

p_enc_3d = PositionalEncoding3D(11)
z = torch.zeros((1,5,6,4,11))
print(p_enc_3d(z).shape) # (1, 5, 6, 4, 11)
```

And for tensors of the form `(batchsize, ch, x)` or their 2D and 3D
counterparts, include the word `Permute` before the number in the class; e.g.
for a 1D input of size `(batchsize, ch, x)`, do `PositionalEncodingPermute1D`
instead of `PositionalEncoding1D`.


```python3
import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute3D

p_enc_3d = PositionalEncodingPermute3D(11)
z = torch.zeros((1,11,5,6,4))
print(p_enc_3d(z).shape) # (1, 11, 5, 6, 4)
```

### Tensorflow Keras

This also supports Tensorflow. Simply prepend all class names with `TF`.

```python3
import tensorflow as tf
from positional_encodings.tf_encodings import TFPositionalEncoding2D, TFSummer

# Returns the position encoding only
p_enc_2d = TFPositionalEncoding2D(170)
y = tf.zeros((1,8,6,2))
print(p_enc_2d(y).shape) # (1, 8, 6, 2)

# Return the inputs with the position encoding added
add_p_enc_2d = TFSummer(TFPositionalEncoding2D(170))
y = tf.ones((1,8,6,2))
print(add_p_enc_2d(y) - p_enc_2d(y)) # tf.ones((1,8,6,2))
```

## Changes as of version `6.0.1`

Before `6.0.1`, users had to install both the `tensorflow` and the
`torch` packages, both of which are quite large. Now, one can install the
packages individually, but now the code has to be changed:

If using PyTorch:

```
from positional_encodings import * -> from positional_encodings.torch_encodings import *
```

If using TensorFlow:

```
from positional_encodings import * -> from positional_encodings.tf_encodings import *
```

## Formulas

The formula for inserting the positional encoding are as follows:

1D:
```
PE(x,2i) = sin(x/10000^(2i/D))
PE(x,2i+1) = cos(x/10000^(2i/D))

Where:
x is a point in 2d space
i is an integer in [0, D/2), where D is the size of the ch dimension
```

2D:
```
PE(x,y,2i) = sin(x/10000^(4i/D))
PE(x,y,2i+1) = cos(x/10000^(4i/D))
PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))

Where:
(x,y) is a point in 2d space
i,j is an integer in [0, D/4), where D is the size of the ch dimension
```

3D:
```
PE(x,y,z,2i) = sin(x/10000^(6i/D))
PE(x,y,z,2i+1) = cos(x/10000^(6i/D))
PE(x,y,z,2j+D/3) = sin(y/10000^(6j/D))
PE(x,y,z,2j+1+D/3) = cos(y/10000^(6j/D))
PE(x,y,z,2k+2D/3) = sin(z/10000^(6k/D))
PE(x,y,z,2k+1+2D/3) = cos(z/10000^(6k/D))

Where:
(x,y,z) is a point in 3d space
i,j,k is an integer in [0, D/6), where D is the size of the ch dimension
```

The 3D formula is just a natural extension of the 2D positional encoding used
in [this](https://arxiv.org/pdf/1908.11415.pdf) paper.

Don't worry if the input is not divisible by 2 (1D), 4 (2D), or 6 (3D); all the
necessary padding will be taken care of.

## Thank you

Thank you for [this](https://github.com/wzlxjtu/PositionalEncoding2D) repo for inspriration of this method.

## Citations
1D:
```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```

2D:
```bibtex
@misc{wang2019translating,
    title={Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training},
    author={Zelun Wang and Jyh-Charn Liu},
    year={2019},
    eprint={1908.11415},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

3D:
Coming soon!
