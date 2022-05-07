#! /bin/bash
sudo docker run -it --network host -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib -e PYTHONPATH=/app -e TF_CPP_MIN_LOG_LEVEL=3 -v "$(pwd)":/app  --gpus all tc-dev
