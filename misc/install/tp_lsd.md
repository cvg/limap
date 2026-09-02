## TP-LSD installation

```bash
git clone --recursive https://github.com/rpautrat/TP-LSD.git ./src/thirdparty/TP-LSD
python -m pip install -e ./src/thirdparty/TP-LSD/tp_lsd/modeling/DCNv2
python -m pip install -e ./src/thirdparty/TP-LSD
```

The TP-LSD Python package is tested under gcc-8 and gcc-9. However, it is known that gcc-10 is currently not supported for the installation of TP-LSD.

The implementation of TP-LSD is originated from [https://github.com/Siyuada7/TP-LSD](https://github.com/Siyuada7/TP-LSD) and later adapted with pip installation by [Rémi Pautrat](https://github.com/rpautrat) in his [forked repo](https://github.com/rpautrat/TP-LSD).

