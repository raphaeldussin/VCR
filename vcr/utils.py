import xarray as xr
import numpy as np


def bounds_2d_to_1d(bnds, dim='nbounds'):
    """ convert 2d bounds xarray.DataArray bnds(ndepth, nbounds)
        into 1d np.array bnds(ndepth+1)
    """

    out = np.concatenate([[bnds.isel({dim:0})[0]], bnds.isel({dim:1})], axis=0)
    return out
