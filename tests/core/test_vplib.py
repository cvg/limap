import numpy as np
import numpy.testing as npt
import pytest

import limap
from limap.image.groups.vplib import (
    VPResult,
    convert_vpresult_to_groups2d,
    convert_vpresults_to_groups2d,
)


def _make_vpresult():
    # Four lines: 0 and 3 on the first VP, 1 on the second, 2 unassigned.
    vps = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    return VPResult([0, 1, -1, 0], vps)


@pytest.mark.ci_workflow
def test_convert_vpresult_to_groups2d():
    groups = convert_vpresult_to_groups2d(_make_vpresult())
    assert len(groups) == 2
    assert all(g.type == limap.geometry.GroupType.VP for g in groups)
    npt.assert_allclose(groups[0].params, np.array([1.0, 0.0, 0.0]))
    assert [line.idx for line in groups[0].lines] == [0, 3]
    assert [line.idx for line in groups[1].lines] == [1]


@pytest.mark.ci_workflow
def test_convert_vpresults_to_groups2d_accepts_dict():
    # The VP detectors are pure Python and hand over a plain dict, so the
    # binding has to accept one (and hand a dict back) regardless of the
    # colmap hash map backend the extension was built against.
    vpresults = {0: _make_vpresult(), 1: VPResult()}
    all_groups2d = convert_vpresults_to_groups2d(vpresults)
    assert isinstance(all_groups2d, dict)
    assert set(all_groups2d) == {0, 1}
    assert len(all_groups2d[0]) == 2
    assert len(all_groups2d[1]) == 0
