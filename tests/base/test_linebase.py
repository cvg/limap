import pytest

import limap
import numpy as np
import numpy.testing as npt

@pytest.mark.ci_workflow
def test_line2d():
    line = limap.base.Line2d(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    npt.assert_allclose(line.length(), np.sqrt(2), rtol=1e-5, atol=1e-8)
    npt.assert_allclose(line.direction(), np.array([1., 1.]) / line.length(), rtol=1e-5, atol=1e-8)

