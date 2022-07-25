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
echo "sphinx check"
cd docs && sphinx-build source build/html && sphinx-build source -D language="zh" build/html_cn
echo "all checks passed, congratulation! 💐"
