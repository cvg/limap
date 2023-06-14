python -m pip install -Ive ..
touch index.rst
touch tutorials/*
touch api/limap.base.*
touch api/limap.estimators.*
touch api/limap.evaluation.*
# touch api/*.rst
make html
