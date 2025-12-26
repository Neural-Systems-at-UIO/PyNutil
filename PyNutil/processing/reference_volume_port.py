import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NearestNDInterpolator:
    """kNN interpolator over scattered points.

    This is a small, dependency-light port of the reference code's behavior.
    It uses a KDTree and supports k-nearest averaging.
    """

    points: np.ndarray  # (N, D)
    values: np.ndarray  # (N,)

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float32)
        vals = np.asarray(self.values)
        if pts.ndim != 2:
            raise ValueError(f"points must be 2D; got {pts.shape}")
        if vals.ndim != 1:
            vals = vals.reshape(-1)
        if pts.shape[0] != vals.shape[0]:
            raise ValueError(
                f"points and values length mismatch: {pts.shape[0]} vs {vals.shape[0]}"
            )

        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("SciPy is required for NearestNDInterpolator") from exc

        self.points = pts
        self.values = vals.astype(np.float32, copy=False)
        self._tree = cKDTree(self.points)

    def __call__(self, query_points: np.ndarray, *, k: int = 1) -> np.ndarray:
        if k < 1:
            raise ValueError("k must be >= 1")
        q = np.asarray(query_points, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.points.shape[1]:
            raise ValueError(
                f"query_points must be (M,{self.points.shape[1]}); got {q.shape}"
            )

        dist, ind = self._tree.query(q, k=k)
        if k == 1:
            return self.values[ind]

        neigh = self.values[ind]
        return neigh.mean(axis=1)


def interpolate(gv: np.ndarray, fv: np.ndarray, *, k: int, resolution: float) -> np.ndarray:
    """Reference-style interpolation.

    Port of the provided `interpolate(gv, fv, k, resolution)`:
    - Chooses ccfv3augmented atlas at 10um or 25um based on resolution
    - Builds a target mask by nearest-neighbor resampling (zoom order=0)
    - Fits kNN on voxels with data inside the mask, then evaluates inside the mask

    Args:
        gv: signal volume
        fv: frequency volume
        k: number of nearest neighbors
        resolution: voxel size (microns)

    Returns:
        Interpolated signal volume (same shape as gv)
    """

    try:
        import brainglobe_atlasapi  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("brainglobe-atlasapi is required for reference interpolate()") from exc

    try:
        from scipy.ndimage import zoom  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SciPy is required for reference interpolate()") from exc

    if resolution < 25:
        atlas = brainglobe_atlasapi.BrainGlobeAtlas("ccfv3augmented_mouse_10um").annotation
        atlas_res = 10
    else:
        atlas = brainglobe_atlasapi.BrainGlobeAtlas("ccfv3augmented_mouse_25um").annotation
        atlas_res = 25

    # Reorient to match the reference convention.
    atlas = np.transpose(atlas, [2, 0, 1])[::-1, ::-1, ::-1]

    atlas_sh = np.array(atlas.shape, dtype=np.float32)
    tgt_sh = (atlas_sh * (float(atlas_res) / float(resolution))).astype(int)

    # Resample atlas to target grid with nearest-neighbor.
    sf = tgt_sh / atlas_sh
    at = zoom(atlas, sf, order=0)

    mask = at != 0
    valid = np.asarray(fv) != 0
    fit = mask & valid

    # Build a full grid in ijk order.
    grid = np.mgrid[0 : tgt_sh[0], 0 : tgt_sh[1], 0 : tgt_sh[2]]
    pts = grid.reshape((3, -1)).T

    nn1 = NearestNDInterpolator(pts[fit.flatten()], np.asarray(gv)[fit])
    out = np.zeros_like(gv, dtype=np.float32)
    out[mask] = nn1(pts[mask.flatten()], k=k)
    return out


def write_nifti(
    volume: np.ndarray,
    resolution: float,
    output_path: str,
    origin_offsets: Optional[np.ndarray] = None,
) -> None:
    """Reference NIfTI writer (Siibra compatible, microns)."""

    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("nibabel is required for write_nifti") from exc

    if origin_offsets is None:
        origin_offsets = np.array([0, 0, 0], dtype=np.float32)

    dims = np.array(volume.shape, dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] *= float(resolution)
    affine[:3, 3] = -0.5 * dims * float(resolution) + origin_offsets

    img = nib.Nifti1Image(np.asarray(volume, dtype=np.uint8), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header["xyzt_units"] = 3

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path + ".nii.gz")


# ----------------------------
# Additional helper functions
# ----------------------------


def find_plane_equation(alignment: np.ndarray):
    v1 = np.asarray(alignment[:3], dtype=np.float32)
    v2 = np.asarray(alignment[3:6], dtype=np.float32)
    v3 = np.asarray(alignment[6:9], dtype=np.float32)
    n = np.cross(v2, v3)
    k = -float(np.dot(n, v1))
    return (float(n[0]), float(n[1]), float(n[2])), float(k)


def _rot_x(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def section_adjust(alignment: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """Rotate the basis vectors of an alignment around a named axis."""

    a = np.asarray(alignment, dtype=np.float32).copy()
    v1 = a[:3]
    v2 = a[3:6]
    v3 = a[6:9]

    if axis.upper() == "ML":
        R = _rot_z(angle)
    elif axis.upper() == "DV":
        R = _rot_x(angle)
    else:
        R = _rot_y(angle)

    v2r = (R @ v2.reshape(3, 1)).reshape(3)
    v3r = (R @ v3.reshape(3, 1)).reshape(3)

    out = np.concatenate([v1, v2r, v3r]).astype(np.float32, copy=False)
    return out


def get_angle_from_alignment(alignment: np.ndarray):
    """Compute approximate DV/ML angles from an alignment.

    This is a lightweight substitute for the original helper.
    The returned angles are used by `generate_square_alignment`.
    """

    v2 = np.asarray(alignment[3:6], dtype=np.float32)
    v3 = np.asarray(alignment[6:9], dtype=np.float32)

    # Use the normal to estimate tilt.
    n = np.cross(v2, v3)
    n_norm = float(np.linalg.norm(n))
    if n_norm == 0:
        return 0.0, 0.0
    n = n / n_norm

    # Heuristic: DV tilt from y/z of normal, ML from x/y of normal.
    dv_angle = float(np.arctan2(n[2], n[1]))
    ml_angle = float(np.arctan2(n[0], n[1]))
    return dv_angle, ml_angle


def generate_square_alignment(anchoring, atlas_shape, resolution):
    basic_alignment = np.array([0, 0, 0, atlas_shape[0], 0, 0, 0, 0, atlas_shape[2]], dtype=np.float32)
    DVangle, MLangle = get_angle_from_alignment(anchoring)
    basic_alignment = section_adjust(basic_alignment, "ML", MLangle)
    basic_alignment = section_adjust(basic_alignment, "DV", DVangle)
    (_, cy, _), k = find_plane_equation(anchoring)
    if cy == 0:
        y_pos = 0.0
    else:
        y_pos = -k / cy
    temp_alignment = basic_alignment.copy()
    temp_alignment[1] = y_pos
    # this is the offset for bbp
    temp_alignment[1] += 24 * (25 / resolution)
    return temp_alignment


def generate_rectangular_image(img, pts):
    import cv2

    width = max(
        np.sqrt(((pts[0][0] - pts[1][0]) ** 2) + ((pts[0][1] - pts[1][1]) ** 2)),
        np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2)),
    )
    height = max(
        np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2)),
        np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2)),
    )
    output_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = pts.astype(np.float32)
    M = cv2.getPerspectiveTransform(pts, output_pts)
    output_img = cv2.warpPerspective(img, M, (int(width), int(height)))
    return output_img


def generate_padded_image(img, pts):
    import cv2

    pts = np.array(pts)
    imheight, imwidth = img.shape

    left_pad = int(np.abs(np.min((0, np.min(pts[:, 0])))))
    right_pad = int(np.max((0, np.max(pts[:, 0]) - (imwidth - left_pad))))

    top_pad = int(np.abs(np.min((0, np.min(pts[:, 1])))))
    bottom_pad = int(np.max((0, np.max(pts[:, 1]) - (imheight - top_pad))))

    img = cv2.copyMakeBorder(
        img,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    adjusted_pts = pts.copy()
    adjusted_pts[:, 0] += left_pad
    adjusted_pts[:, 1] += top_pad

    return img, adjusted_pts


def find_combination(alignment, target):
    v1 = np.asarray(alignment[:3], dtype=np.float32)
    v2 = np.asarray(alignment[3:6], dtype=np.float32)
    v3 = np.asarray(alignment[6:], dtype=np.float32)

    target = np.asarray(target, dtype=np.float32)

    A = np.vstack([v2, v3]).T
    b = target - v1

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M, N = x
    return float(M), float(N)


def find_vector(data, dest="Zero"):
    # Analytic port of the original sympy-based solver.
    data = np.asarray(data, dtype=np.float32)
    a = data[3:6]
    b = data[6:9]

    if dest == "Zero":
        target_ratio = float(data[0] / data[2])
        # ((a0*x + b0)/2) / ((a2*x + b2)/2) = target_ratio
        denom = float(a[0] - target_ratio * a[2])
        if denom == 0:
            x = 0.0
        else:
            x = float((target_ratio * b[2] - b[0]) / denom)
        vector = 0.5 * (a * x + b)
        magnitude = np.mean(np.array([data[0], data[2]], dtype=np.float32) / vector[[0, 2]])
        return x, vector, float(magnitude)

    if dest == "X":
        # ((a2*x + b2)/2) / ((b2*x + a2)/2) ??? original code seems inconsistent;
        # keep a simple fallback.
        x = 0.0
        vector = 0.5 * (a * x + b)
        magnitude = 1.0 / float(vector[0]) if vector[0] != 0 else 0.0
        return x, vector, float(magnitude)

    if dest == "Z":
        target_ratio = float(data[0] / data[2])
        denom = float(a[0] - target_ratio * a[2])
        if denom == 0:
            x = 0.0
        else:
            x = float((target_ratio * b[2] - b[0]) / denom)
        vector = 0.5 * (a * x + b)
        magnitude = 1.0 / float(vector[2]) if vector[2] != 0 else 0.0
        return x, vector, float(magnitude)

    x = 0.0
    vector = 0.5 * (a * x + b)
    return x, vector, 0.0


def perfect_image(img, alignment, resolution=25):
    import cv2

    # These constants come from the reference implementation.
    x_size = 11400 / resolution
    y_size = 8000 / resolution

    (cx, cy, cz), k = find_plane_equation(alignment)

    top_leftX, top_leftZ = 0, 0
    top_leftY = -(cx * top_leftX + cy + top_leftZ * cz + k) / cy
    top_left = (top_leftX, top_leftY, top_leftZ)

    top_rightX, top_rightZ = x_size, 0
    top_rightY = -(cx * top_rightX + cy + top_rightZ * cz + k) / cy
    top_right = (top_rightX, top_rightY, top_rightZ)

    bottom_rightX, bottom_rightZ = x_size, y_size
    bottom_rightY = -(cx * bottom_rightX + cy + bottom_rightZ * cz + k) / cy
    bottom_right = (bottom_rightX, bottom_rightY, bottom_rightZ)

    bottom_leftX, bottom_leftZ = 0, y_size
    bottom_leftY = -(cx * bottom_leftX + cy + bottom_leftZ * cz + k) / cy
    bottom_left = (bottom_leftX, bottom_leftY, bottom_leftZ)

    size = (
        np.array((np.linalg.norm(alignment[3:6]), np.linalg.norm(alignment[6:9])))
        .round()
        .astype(int)
    )
    img = cv2.resize(img, tuple(int(x) for x in size))
    imheight, imwidth = img.shape

    tlM, tlN = find_combination(alignment, top_left)
    tlX = int(np.round(tlM * imwidth))
    tlY = int(np.round(tlN * imheight))

    trM, trN = find_combination(alignment, top_right)
    trX = int(np.round(trM * imwidth))
    trY = int(np.round(trN * imheight))

    brM, brN = find_combination(alignment, bottom_right)
    brX = int(np.round(brM * imwidth))
    brY = int(np.round(brN * imheight))

    blM, blN = find_combination(alignment, bottom_left)
    blX = int(np.round(blM * imwidth))
    blY = int(np.round(blN * imheight))

    pts = np.array([(tlX, tlY), (trX, trY), (brX, brY), (blX, blY)])
    pad_img, pts = generate_padded_image(img, pts)
    output_img = generate_rectangular_image(pad_img, pts)
    return output_img, pts


def generate_target_coordinates(alignment: np.ndarray, atlas_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate voxel indices for a target slice.

    Returns a tuple (x_idx, y_idx, z_idx) suitable for np.add.at.
    """

    a = np.asarray(alignment, dtype=np.float32)
    v1 = a[:3]
    v2 = a[3:6]
    v3 = a[6:9]

    sx, sy, sz = (int(atlas_shape[0]), int(atlas_shape[1]), int(atlas_shape[2]))

    # Create a grid over X and Z.
    xs = np.arange(sx, dtype=np.float32)
    zs = np.arange(sz, dtype=np.float32)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")

    # Normalized coordinates in [0,1]
    mx = xx / max(sx - 1, 1)
    nz = zz / max(sz - 1, 1)

    pts = v1.reshape(1, 1, 3) + mx[..., None] * v2.reshape(1, 1, 3) + nz[..., None] * v3.reshape(1, 1, 3)

    xi = np.rint(pts[..., 0]).astype(np.int64)
    yi = np.rint(pts[..., 1]).astype(np.int64)
    zi = np.rint(pts[..., 2]).astype(np.int64)

    xi = np.clip(xi, 0, sx - 1)
    yi = np.clip(yi, 0, sy - 1)
    zi = np.clip(zi, 0, sz - 1)

    return xi.reshape(-1), yi.reshape(-1), zi.reshape(-1)
