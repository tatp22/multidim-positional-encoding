# 1D, 2D, and 3D Sinusodal Postional Encoding Pytorch

This is an implemenation of 1D, 2D, and 3D sinusodal positional encoding, being able to encode on tensors of the form `(batchsize, x, ch)`, `(batchsize, x, y, ch)`, and `(batchsize, x, y, z, ch)`, where the positional encodings will be added to the `ch` dimension. The [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) allowed for positional encoding in only one dimension, however, this works to extend this to 2 and 3 dimensions.

To install, simply run:

```
pip install positional-encodings
```

Specifically, the formula for inserting the positional encoding will be as follows:

1D:
```
PE(x,2i) = sin(x/10000^(2i/D))
PE(x,2i+1) = cos(x/10000^(2i/D))

Where:
x is a point in 2d space
i is in [0, D/2), where D is the size of the ch dimension
```

2D:
```
PE(x,y,2i) = sin(x/10000^(4i/D))
PE(x,y,2i+1) = cos(x/10000^(4i/D))
PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))

Where:
(x,y) is a point in 2d space
i,j is in [0, D/4), where D is the size of the ch dimension
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
i,j,k is in [0, D/6), where D is the size of the ch dimension
```

This is just a natural extension of the 3D positional encoding used in [this](https://arxiv.org/pdf/1908.11415.pdf) paper.

Don't worry if the input is not divisible by 2 (1D), 4 (2D), or 6 (3D); all the necessary padding will be taken care of.

## Usage:

```python3
import torch
from pos_enc_multidim import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D

p_enc_1d = PositionalEncoding1D(10)
x = torch.zeros((1,6,10))
print(p_enc_1d(x).shape) # (1, 6, 10)

p_enc_2d = PositionalEncoding2D(8)
y = torch.zeros((1,6,2,8))
print(p_enc_2d(y).shape) # (1, 6, 2, 8)

p_enc_3d = PositionalEncoding3D(11)
z = torch.zeros((1,5,6,4,11))
print(p_enc_3d(z).shape) # (1, 5, 6, 4, 11)
```

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
