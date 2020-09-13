# Sparse-Matrix_CUDA

Installation CUDA in your notebook

## Check Python Version (Python 3.6.9)
!python --version

## Check Ubuntu Version (Ubuntu 18.04.3 LTS)
!lsb_release -a

## Check CUDA/cuDNN Version (Built on Sun_Jul_28_19:07:16_PDT_2019 / Cuda compilation tools, release 10.1, V10.1.243)
!nvcc -V && which nvcc

## Check GPU
!nvidia-smi

## Install RAPIDS
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git <br />
!bash rapidsai-csp-utils/colab/rapids-colab.sh stable <br />

import sys, os  <br /><br />
dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages') <br />
sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:] <br />
sys.path  <br />
exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

## Try on this example
from cupy.sparse import random  <br /><br />
S = random(100000, 2000, density=6e-2)  <br />
S.get()
