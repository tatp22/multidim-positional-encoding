import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="positional_encodings",
    version="5.0.0",
    author="Peter Tatkowski",
    author_email="tatp22@gmail.com",
    description="1D, 2D, and 3D Sinusodal Positional Encodings in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tatp22/multidim-positional-encoding",
    packages=setuptools.find_packages(),
    keywords=['transformers', 'attention'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'tensorflow',
        'numpy',
    ],

)
