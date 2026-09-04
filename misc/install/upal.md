## UPAL installation

UPAL needs two packages: the network itself, and the point-seeded LSD extension
its line detection is built on. The network predicts keypoints, descriptors and
a line distance field, but no segments -- those come from LSD seeded by the
keypoints and filtered by the field.

```bash
python -m pip install "upal-local-features@git+https://github.com/francois141/upal.git@91d79a0"
python -m pip install points-lsd
```

The modules ``upal`` and ``points_lsd`` should be available afterwards. UPAL is
not on PyPI, and it does not declare ``points-lsd`` as a dependency, so both
lines are needed.

``points_lsd`` does not conflict with the ``pytlsd`` that `requirements.txt`
installs: the module names differ and their ``lsd()`` outputs are identical.
Do **not** follow the upstream README's `third_party/points_lsd` submodule
build instead -- at the pinned commit it builds a module named ``pytlsd`` and
would overwrite limap's.

The checkpoint downloads automatically on first use; no manual download needed.
UPAL runs on CPU but is only worth using on a GPU.
