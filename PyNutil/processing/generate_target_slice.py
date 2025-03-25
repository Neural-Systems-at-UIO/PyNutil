import numpy as np
import math


def generate_target_slice(ouv, atlas):
    """
    Generate a 2D slice from a 3D atlas based on orientation vectors.

    Args:
        ouv (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        atlas (ndarray): 3D atlas volume.

    Returns:
        ndarray: 2D slice extracted from the atlas.
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = ouv
    width = np.floor(math.hypot(ux, uy, uz)).astype(int) + 1
    height = np.floor(math.hypot(vx, vy, vz)).astype(int) + 1
    data = np.zeros((width, height), dtype=np.uint32).flatten()
    xdim, ydim, zdim = atlas.shape

    y_values = np.arange(height)
    x_values = np.arange(width)

    hx = ox + vx * (y_values / height)
    hy = oy + vy * (y_values / height)
    hz = oz + vz * (y_values / height)

    wx = ux * (x_values / width)
    wy = uy * (x_values / width)
    wz = uz * (x_values / width)

    lx = np.floor(hx[:, None] + wx).astype(int)
    ly = np.floor(hy[:, None] + wy).astype(int)
    lz = np.floor(hz[:, None] + wz).astype(int)

    valid_indices = (
        (0 <= lx) & (lx < xdim) & (0 <= ly) & (ly < ydim) & (0 <= lz) & (lz < zdim)
    ).flatten()

    lxf = lx.flatten()
    lyf = ly.flatten()
    lzf = lz.flatten()

    valid_lx = lxf[valid_indices]
    valid_ly = lyf[valid_indices]
    valid_lz = lzf[valid_indices]

    atlas_slice = atlas[valid_lx, valid_ly, valid_lz]
    data[valid_indices] = atlas_slice

    data_im = data.reshape((height, width))
    return data_im
