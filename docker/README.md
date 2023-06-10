Run the following command to build the docker for tensorcircuit at parent path:

```bash
sudo docker build . -f docker/Dockerfile -t tensorcircuit
```

Since v0.10 we introduce new docker env based on ubuntu20.04+cuda11.7+py3.10 (+ pip installed tensorcircuit package), build the new docker use

```bash
sudo docker build . -f docker/Dockerfile_v2 -t tensorcircuit
```

One can also pull the [official image](https://hub.docker.com/repository/docker/tensorcircuit/tensorcircuit) from DockerHub as

```bash
sudo docker pull tensorcircuit/tensorcircuit
```

Run the docker container by the following command:

```bash
sudo docker run -it --network host --gpus all tensorcircuit

# if one also wants to mount local source code, also add args `-v "$(pwd)":/root`

# using tensorcircuit/tensorcircuit:latest to run the prebuild docker image from dockerhub
```

`export CUDA_VISIBLE_DEVICES=-1` if you want to test only on CPU.
