# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
# FROM pytorch/pytorch:0.4_cuda9_cudnn7
# FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
FROM nvcr.io/nvidia/pytorch:21.12-py3

LABEL version="0.5"

# USER root

ENV DEBIAN_FRONTEND noninteractive

# build essentials
# RUN apt-get update && apt-get -y install cmake
RUN apt-get update && apt-get install -y vim
RUN apt-get install -y build-essential
RUN apt-get update && apt-get install -y git-all

# python essentials
# Starting from numpy 1.24.0 np.float would pop up an error. 
# Cocoeval tool still uses np.float. As a result we can only use numpy 1.23.0.
# Alternatively, we can clone the cocoapi and modify the cocoeval tool.
# https://github.com/cocodataset/cocoapi/pull/569
RUN pip install numpy==1.23.0
RUN pip install -U matplotlib
RUN pip install scipy
RUN pip install tensorboardX
RUN pip install wandb

# torch
RUN pip install torch>=1.7.0
RUN pip install torchvision>=0.8.1


RUN pip install einops

