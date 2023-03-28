FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md

RUN apt-key del 7fa2af80
RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Nvidia is breaking the docker images on Apr 2022 ...
# https://github.com/NVIDIA/nvidia-docker/issues/1631

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
    wget \
    cuda-command-line-tools-11-0 \
    git \
    vim \
    pandoc

RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda install -y  \
    cudatoolkit=11.0 \
    pip \
    python=3.8

COPY requirements/requirements-docker.txt /app/requirements-docker.txt

RUN pip install -r /app/requirements-docker.txt
  
RUN pip install jaxlib==0.3.2+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install -U git+https://github.com/jcmgray/cotengra.git

RUN pip install ray

# requirements conflict for ray

RUN ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11

RUN echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib >> ~/.bashrc \
    && echo export PYTHONPATH=/app >> ~/.bashrc \
    && echo export TF_CPP_MIN_LOG_LEVEL=3 >> ~/.bashrc

COPY . /app

# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/targets/x86_64-linux/lib

WORKDIR /app

CMD ["/bin/bash"]