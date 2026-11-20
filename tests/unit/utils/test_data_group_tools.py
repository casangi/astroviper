"""Tests of utils.data_group_tools."""

import xarray as xr

from astroviper.utils.data_group_tools import remove_data_group_xds


def test_remove_data_group_keeps_the_variables():
    xds = xr.Dataset(
        {"VISIBILITY": ("time", [1.0]), "VISIBILITY_LINE": ("time", [2.0])},
        attrs={
            "data_groups": {
                "base": {"correlated_data": "VISIBILITY"},
                "line": {"correlated_data": "VISIBILITY_LINE"},
            }
        },
    )
    assert remove_data_group_xds(xds, "line") == {"correlated_data": "VISIBILITY_LINE"}
    assert list(xds.attrs["data_groups"]) == ["base"]
    assert "VISIBILITY_LINE" in xds.data_vars
    assert remove_data_group_xds(xds, "line") is None
    assert remove_data_group_xds(xr.Dataset(), "line") is None
