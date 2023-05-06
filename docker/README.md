Run the following command to build the docker for tensorcircuit at parent path:

```bash
sudo docker build . -f docker/Dockerfile -t tensorcircuit
```

One can also pull the [official image](https://hub.docker.com/repository/docker/tensorcircuit/tensorcircuit) from DockerHub as

```bash
sudo docker pull tensorcircuit/tensorcircuit
```

Run the docker container by the following command:

```bash
sudo docker run -it --network host --gpus all tensorcircuit

# if one also wants to mount local source code, also add args `-v "$(pwd)":/app`

# using tensorcircuit/tensorcircuit to run the prebuild docker image from dockerhub

# for old dockerfile with no runtime env setting
# sudo docker run -it --network host -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib -e PYTHONPATH=/app -v "$(pwd)":/app  --gpus all tensorcircuit
```

`export TF_CPP_MIN_LOG_LEVEL=3` maybe necessary since jax suprisingly frequently complain about ptxas version problem. And `export CUDA_VISIBLE_DEVICES=-1` if you want to test only on CPU.

The built docker has no tensorcircuit pip package installed but left with a tensorcircuit source code dir. So one can `python setup.py develop` to install tensorcircuit locally (one can also mount the tensorcircuit codebase on host) or `pip install tensorcircuit` within the running docker.
