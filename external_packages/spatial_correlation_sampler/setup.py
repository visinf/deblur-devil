#!/usr/bin/env python3

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

CPU_ONLY = False

sources = ['correlation.cpp', 'correlation_sampler.cpp']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_75,code=compute_75',
    '-O3'
]

if CPU_ONLY:
    Extension = CppExtension
    macro = []
else:
    Extension = CUDAExtension
    sources.append('correlation_cuda_kernel.cu')
    macro = [("USE_CUDA", None)]

setup(
    name='spatial_correlation_sampler',
    version="0.2.0",
    author="Clément Pinard",
    author_email="clement.pinard@ensta-paristech.fr",
    description="Correlation module for pytorch",
    long_description_content_type="text/markdown",
    url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
    install_requires=['torch>=1.1', 'numpy'],
    ext_modules=[
        Extension('spatial_correlation_sampler_backend',
                  sources,
                  define_macros=macro,
                  extra_compile_args={'cxx': ['-fopenmp'], 'nvcc': nvcc_args},
                  extra_link_args=['-lgomp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ])
