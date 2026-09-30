## LightGlueStick installation

```bash
python -m pip install --no-deps \
    "lightgluestick@git+https://github.com/aubingazhib/LightGlueStick.git@98630b9"
```

`--no-deps` is required: upstream pins `numpy<2` and `opencv-python==4.7.0.*`,
which would downgrade both out from under LIMAP. Nothing it needs is missing
from a LIMAP environment. The checkpoint downloads automatically on first use.
