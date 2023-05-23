#!/bin/bash

# conda create -n "nicp" python=3.9
conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install -y pytorch3d -c pytorch3d
conda install -y cuda-toolkit=11.6
pip install numpy scikit-learn open3d==0.14.1 matplotlib face_alignment