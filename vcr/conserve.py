import numpy as np
from numba import njit


class layer():
    """ vertical layer properties """
    def __init__(self, mindepth, maxdepth):
        self.mindepth = mindepth
        self.maxdepth = maxdepth
        self.thickness = maxdepth - mindepth


def create_remapping_matrix(depth_bnds_src, depth_bnds_tgt, strict=False):
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

    # this add the residual to the last non-zero cell so that sum of weights
    # add up to one.
    if not strict:
        # find the index of incomplete bottom cell and compute residual
        total_tgt = remap_matrix.sum(axis=1)
        inv_total_tgt = total_tgt[::-1]
        kbottom_tgt = ntgt - 1  # init to the last cell
        residual = 0.
        for kz in range(len(inv_total_tgt)):
            if inv_total_tgt[kz] != 0.:  # don't want cells deeper than source
                if inv_total_tgt[kz] != 1.:  # don't want complete cells
                    kbottom_tgt = ntgt - 1 - kz
                    residual = 1 - total_tgt[kbottom_tgt]
                    break

        # find the index of the last source cell used
        weights_bottom_tgt = remap_matrix[kbottom_tgt, :].squeeze()
        inv_weights_bottom_tgt = weights_bottom_tgt[::-1]
        kbottom_src = nsrc - 1  # init to the last cell
        for kz in range(len(inv_weights_bottom_tgt)):
            if inv_weights_bottom_tgt[kz] != 0.:
                kbottom_src = nsrc - 1 - kz
                break

        remap_matrix[kbottom_tgt, kbottom_src] += residual

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

    # first guess
    datain[np.isnan(datain)] = 0
    datainT = np.transpose(datain, axes=[1, 0, 2])
    out = np.matmul(remap_matrix, datainT)
    out = np.transpose(out, axes=[1, 0, 2])
    out = np.ma.masked_values(out, 0)

    # correct for bathy
    out = add_bathy_mismatch(datain, out, remap_matrix)

    return out


@njit
def add_bathy_mismatch(data_src, data_tgt, remap_matrix):
    """ correct for incomplete remapping due to topography """
    nz_src, ny, nx = data_src.shape
    nz_tgt, ny, nx = data_tgt.shape
    for ky in range(ny):
        for kx in range(nx):
            # find bottom cell on source grid: this is one cell above
            # first zero point in the data
            column_src = data_src[:, ky, kx]
            bottom_src = np.abs(column_src).argmin() - 1

            # compute the residual for this particular bottom level:
            # 1 - sum of coef between surface and bottom level
            residual = 1 - remap_matrix[:, :bottom_src+1].sum(axis=1)
            # filter out zeros (account for num repres error)
            tmp = residual.copy()
            tmp[np.where(tmp <= 1e-12)] = 1e+20
            # bottom level on target grid is the only non-zero (1e+20)
            # or non equal to one value
            bottom_tgt = np.abs(tmp).argmin()
            # get value of residual
            residual_bottom = residual[bottom_tgt]

            # if residual found is one, then no correction is needed
            if residual_bottom < 1.:
                # add to the target bottom level the residual * input data
                data_tgt[bottom_tgt, ky, kx] += residual_bottom * \
                                                data_src[bottom_src, ky, kx]

    return data_tgt

#@njit
#def correct_bottom(array, depth):
#    """ in conservative remapping bottom cells are likely to have inconsistent
#    thickness between source and target grid.
#    Switch to linear extrapolation for that cell """
#    nz, ny, nx = array.shape
#    for ky in range(ny):
#        for kx in range(nx):
#            column = array[:, ky, kx]
#            idx_bottom = np.abs(column).argmin()
#            if idx_bottom > 1:
#                # linear extrapolation
#                dz1 = depth[idx_bottom] - depth[idx_bottom-1]
#                dz2 = depth[idx_bottom-1] - depth[idx_bottom-2]
#                scale = dz1/dz2
#                update = (1 + scale) * column[idx_bottom-1] - \
#                    scale * column[idx_bottom-2]
#                array[idx_bottom, ky, kx] = update
#            elif idx_bottom > 0:
#                # copy value
#                array[idx_bottom, ky, kx] = array[idx_bottom-1, ky, kx]
#    return array
