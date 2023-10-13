import numpy as np


def generate_target_slice(alignment, volume):
    Ox, Oy, Oz, Ux, Uy, Uz, Vx, Vy, Vz = alignment
    ##just for mouse for now
    bounds = [455, 527, 319]
    X_size = np.sqrt(np.sum(np.square((Ux, Uy, Uz))))
    Z_size = np.sqrt(np.sum(np.square((Vx, Vy, Vz))))
    X_size = np.round(X_size).astype(int)
    Z_size = np.round(Z_size).astype(int)
    # make this into a grid (0,0) to (320,456)
    Uarange = np.arange(0, 1, 1 / X_size)
    Varange = np.arange(0, 1, 1 / Z_size)
    Ugrid, Vgrid = np.meshgrid(Uarange, Varange)
    Ugrid_x = Ugrid * Ux
    Ugrid_y = Ugrid * Uy
    Ugrid_z = Ugrid * Uz
    Vgrid_x = Vgrid * Vx
    Vgrid_y = Vgrid * Vy
    Vgrid_z = Vgrid * Vz

    X_Coords = (Ugrid_x + Vgrid_x).flatten() + Ox
    Y_Coords = (Ugrid_y + Vgrid_y).flatten() + Oy
    Z_Coords = (Ugrid_z + Vgrid_z).flatten() + Oz

    X_Coords = np.round(X_Coords).astype(int)
    Y_Coords = np.round(Y_Coords).astype(int)
    Z_Coords = np.round(Z_Coords).astype(int)

    out_bounds_Coords = (
        (X_Coords > bounds[0])
        | (Y_Coords > bounds[1])
        | (Z_Coords > bounds[2])
        | (X_Coords < 0)
        | (Y_Coords < 0)
        | (Z_Coords < 0)
    )
    X_pad = X_Coords.copy()
    Y_pad = Y_Coords.copy()
    Z_pad = Z_Coords.copy()

    X_pad[out_bounds_Coords] = 0
    Y_pad[out_bounds_Coords] = 0
    Z_pad[out_bounds_Coords] = 0

    regions = volume[X_pad, Y_pad, Z_pad]
    ##this is a quick hack to solve rounding errors
    C = len(regions)
    compare = C - X_size * Z_size
    if abs(compare) == X_size:
        if compare > 0:
            Z_size += 1
        if compare < 0:
            Z_size -= 1
    elif abs(C - X_size * Z_size) == Z_size:
        if compare > 0:
            X_size += 1
        if compare < 0:
            X_size -= 1
    elif abs(C - X_size * Z_size) == Z_size + X_size:
        if compare > 0:
            X_size += 1
            Z_size += 1
        if compare < 0:
            X_size -= 1
            Z_size -= 1
    elif abs(C - X_size * Z_size) == Z_size - X_size:
        if compare > 0:
            X_size += 1
            Z_size -= 1
        if compare < 0:
            X_size -= 1
            Z_size += 1
    elif abs(C - X_size * Z_size) == X_size - Z_size:
        if compare > 0:
            X_size -= 1
            Z_size += 1
        if compare < 0:
            X_size += 1
            Z_size -= 1
    regions = regions.reshape((abs(Z_size), abs(X_size)))
    return regions
