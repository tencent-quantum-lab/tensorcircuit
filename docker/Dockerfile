FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
    wget \
    cuda-command-line-tools-11-0 \
    git

RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -y  \
    cudatoolkit=11.0 \
    pip \
    python=3.8

COPY . /app

RUN pip install -r /app/requirements-docker.txt
  
RUN pip install jaxlib==0.1.75+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN pip install -U git+https://github.com/jcmgray/cotengra.git

RUN ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

WORKDIR /app

CMD ["/bin/bash"]