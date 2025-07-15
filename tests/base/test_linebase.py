import numpy as np
import numpy.testing as npt
import pytest

import limap


@pytest.mark.ci_workflow
def test_line2d():
    line = limap.base.Line2d(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    npt.assert_allclose(line.length(), np.sqrt(2), rtol=1e-5, atol=1e-8)
    npt.assert_allclose(
        line.direction(),
        np.array([1.0, 1.0]) / line.length(),
        rtol=1e-5,
        atol=1e-8,
    )
