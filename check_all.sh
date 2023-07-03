#! /bin/sh
set -e 
echo "black check"
black . --check
echo "mypy check"
mypy tensorcircuit
echo "pylint check"
pylint tensorcircuit tests examples/*.py
echo "pytest check"
pytest -n auto --cov=tensorcircuit -vv -W ignore::DeprecationWarning
# for test on gpu machine, please set `export TF_FORCE_GPU_ALLOW_GROWTH=true` for tf
# and `export XLA_PYTHON_CLIENT_PREALLOCATE=false` for jax to avoid OOM in testing
echo "sphinx check"
cd docs && sphinx-build source build/html && sphinx-build source -D language="zh" build/html_cn
echo "all checks passed, congratulation! 💐"
