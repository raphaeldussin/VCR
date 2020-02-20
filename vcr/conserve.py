import numpy as np


class layer():
    """ vertical layer properties """
    def __init__(self, mindepth, maxdepth):
        self.mindepth = mindepth
        self.maxdepth = maxdepth
        self.thickness = maxdepth - mindepth


def create_remapping_matrix(depth_bnds_src, depth_bnds_tgt):
    """ create a remapping matrix of dims NxM from depth_bnds_src (M+1)
        and depth_bnds_tgt (N+1)

    PARAMETERS:
    -----------

    depth_bnds_src: 1d np.ndarray
        layer interfaces on source grid
    depth_bnds_tgt: 1d np.array
        layer interfaces on target grid

    RETURN:
    -------

    remap_matrix: 2d np.ndarray
        remapping weights for one column
    """

    nsrc = len(depth_bnds_src) - 1
    ntgt = len(depth_bnds_tgt) - 1

    remap_matrix = np.empty((ntgt, nsrc))

    # create layers
    layers_src = []
    for k in range(nsrc):
        layers_src.append(layer(depth_bnds_src[k], depth_bnds_src[k+1]))
    layers_tgt = []
    for k in range(ntgt):
        layers_tgt.append(layer(depth_bnds_tgt[k], depth_bnds_tgt[k+1]))

    for ktgt in range(ntgt):
        for ksrc in range(nsrc):
            if layers_src[ksrc].maxdepth <= layers_tgt[ktgt].mindepth:
                remap_matrix[ktgt, ksrc] = 0  # source layer is above
            elif layers_src[ksrc].mindepth >= layers_tgt[ktgt].maxdepth:
                remap_matrix[ktgt, ksrc] = 0  # source layer is below
            else:  # overlap
                # compute bounds of overlap
                dmin = max(layers_src[ksrc].mindepth,
                           layers_tgt[ktgt].mindepth)
                dmax = min(layers_src[ksrc].maxdepth,
                           layers_tgt[ktgt].maxdepth)
                thk = layers_tgt[ktgt].thickness
                remap_matrix[ktgt, ksrc] = (dmax - dmin) / thk

    return remap_matrix


def vertical_remap_z2z(datain, remap_matrix):
    """ Vertical remapping for z to z coordinates: this is the simplest case
    and the same remapping matrix can be used for all water columns

    PARAMETERS:
    -----------

    datain: 3d np.ndarray (nz_src, ny, nx)
        data to remap
    remap_matrix: 2d np.ndarray (nz_tgt, nz_src)

    RETURN:
    -------

    out: 3d np.ndarray
        remapped data
    """
    datain[np.isnan(datain)] = 0
    datainT = np.transpose(datain, axes=[1, 0, 2])
    out = np.matmul(remap_matrix, datainT)
    out = np.transpose(out, axes=[1, 0, 2])
    out = np.ma.masked_values(out, 0)
    return out
