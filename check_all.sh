#! /bin/sh
set -e 
echo "black check"
black . --check
echo "mypy check"
mypy tensorcircuit
echo "pylint check"
pylint tensorcircuit tests
echo "pytest check"
pytest --cov=tensorcircuit -vv
echo "sphinx check"
cd docs && make html
echo "all checks passed, congratulates!"