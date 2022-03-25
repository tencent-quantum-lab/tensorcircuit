Run the following command to build the docker for tensorcircuit at parent path:

```bash
sudo docker build . -f docker/Dockerfile -t tc-dev-jupyter
```

Run the docker container by the following command:

```bash
sudo docker run -it --network host -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib -e PYTHONPATH=/app -v "$(pwd)":/app  --gpus all tc-dev-jupyter
```

To utilize tensorflow + cuda, we need to further run the following command within runtime docker:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib
```

`export TF_CPP_MIN_LOG_LEVEL=3` maybe necessary since jax suprisingly frequently complain about ptxas version problem. And `export CUDA_VISIBLE_DEVICES=-1` if you want to test on CPU.
