"""This code was written by Gergely Csucs, Harry Carey and Rembrandt Bakker"""

import numpy as np


def _classify_triangles(x, y, triangles):
    """Partition *triangles* into keep/remove lists via circumcircle test."""
    found = False
    keep = []
    remove = []
    for triangle in triangles:
        if not found and triangle.contains_point(x, y):
            found = True
        if triangle.incircle(x, y):
            remove.append(triangle)
        else:
            keep.append(triangle)
    return found, keep, remove


def _insert_marker(marker, vertices, edges, triangles):
    """Insert a single marker into the triangulation, returning updated triangle list."""
    x, y = marker[2:4]
    found, keep, remove = _classify_triangles(x, y, triangles)
    if found:
        for triangle in remove:
            triangle.removeedges()
    else:
        keep.extend(remove)

    vcount = len(vertices)
    vertices.append(marker)
    for i in range(vcount - 1):
        for j in range(i + 1, vcount):
            if edges[edgeindex(i, j)] == 1:
                keep.append(Triangle(i, j, vcount, vertices, edges))
    return keep


def triangulate(w, h, markers):
    """
    Triangulates a set of markers.

    Args:
        w (int): Width of the image.
        h (int): Height of the image.
        markers (list): List of markers.

    Returns:
        list: List of triangles.
    """
    vertices = [
        [-0.1 * w, -0.1 * h, -0.1 * w, -0.1 * h],
        [1.1 * w, -0.1 * h, 1.1 * w, -0.1 * h],
        [-0.1 * w, 1.1 * h, -0.1 * w, 1.1 * h],
        [1.1 * w, 1.1 * h, 1.1 * w, 1.1 * h],
    ]

    edges = [0] * ((len(markers) + 4) * (len(markers) + 4 - 1) // 2)
    triangles = [Triangle(0, 1, 2, vertices, edges), Triangle(1, 2, 3, vertices, edges)]
    edges[0] = edges[1] = edges[4] = edges[5] = 2

    for marker in markers:
        triangles = _insert_marker(marker, vertices, edges, triangles)
    return triangles


def transform_vec(triangulation, x, y):
    """
    Transforms a set of points using a triangulation.

    Args:
        triangulation (list): List of triangles.
        x (ndarray): X coordinates of the points.
        y (ndarray): Y coordinates of the points.

    Returns:
        tuple: Transformed coordinates.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xPrime = np.zeros(x.shape, np.float64)
    yPrime = np.zeros(y.shape, np.float64)
    if x.size == 0:
        return (xPrime, yPrime)

    remaining = np.ones(x.shape, dtype=bool)
    for triangle in triangulation:
        if not np.any(remaining):
            break
        rem_idx = np.nonzero(remaining)[0]
        ok, x_ok, y_ok = triangle.intriangle_subset(x[rem_idx], y[rem_idx])
        if np.any(ok):
            hit_idx = rem_idx[ok]
            xPrime[hit_idx] = x_ok
            yPrime[hit_idx] = y_ok
            remaining[hit_idx] = False
    return (xPrime, yPrime)


def forwardtransform_vec(triangulation, x, y):
    """
    Forward transforms a set of points using a triangulation.

    Args:
        triangulation (list): List of triangles.
        x (ndarray): X coordinates of the points.
        y (ndarray): Y coordinates of the points.

    Returns:
        tuple: Transformed coordinates.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xPrime = np.zeros(x.shape, np.float64)
    yPrime = np.zeros(y.shape, np.float64)
    if x.size == 0:
        return (xPrime, yPrime)

    remaining = np.ones(x.shape, dtype=bool)
    for triangle in triangulation:
        if not np.any(remaining):
            break
        rem_idx = np.nonzero(remaining)[0]
        ok, x_ok, y_ok = triangle.inforward_subset(x[rem_idx], y[rem_idx])
        if np.any(ok):
            hit_idx = rem_idx[ok]
            xPrime[hit_idx] = x_ok
            yPrime[hit_idx] = y_ok
            remaining[hit_idx] = False
    return (xPrime, yPrime)


def inv3x3(m):
    """Invert a 3x3 matrix.

    Returns None for singular matrices.
    """
    try:
        return np.linalg.inv(np.asarray(m, dtype=np.float64)).tolist()
    except np.linalg.LinAlgError:
        return None


def distsquare(ax, ay, bx, by):
    """
    Calculates the squared distance between two points.

    Args:
        ax (float): X coordinate of the first point.
        ay (float): Y coordinate of the first point.
        bx (float): X coordinate of the second point.
        by (float): Y coordinate of the second point.

    Returns:
        float: Squared distance between the points.
    """
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by)


def edgeindex(a, b):
    """
    Calculates the index of an edge in the edge list.

    Args:
        a (int): Index of the first vertex.
        b (int): Index of the second vertex.

    Returns:
        int: Index of the edge.
    """
    i = min(a, b)
    j = max(a, b)
    return j * (j - 1) // 2 + i


def _build_decomp(A, B, C, i, j):
    """Build a 3x3 decomposition matrix from vertex coordinates at indices *i*, *j*."""
    ax, ay = A[i], A[j]
    bx, by = B[i], B[j]
    cx, cy = C[i], C[j]
    return inv3x3([[bx - ax, by - ay, 0], [cx - ax, cy - ay, 0], [ax, ay, 1]])


class Triangle:
    def __init__(self, a, b, c, vlist, elist):
        """
        Initializes a triangle.

        Args:
            a (int): Index of the first vertex.
            b (int): Index of the second vertex.
            c (int): Index of the third vertex.
            vlist (list): List of vertices.
            elist (list): List of edges.
        """
        self.A = vlist[a]
        self.B = vlist[b]
        self.C = vlist[c]
        self.elist = elist
        self.edges = [edgeindex(a, b), edgeindex(a, c), edgeindex(b, c)]
        for edge in self.edges:
            elist[edge] += 1
        self.forwarddecomp = np.asarray(
            _build_decomp(self.A, self.B, self.C, 0, 1), dtype=np.float64
        )
        self.decomp = np.asarray(
            _build_decomp(self.A, self.B, self.C, 2, 3), dtype=np.float64
        )
        self._compute_circumcircle()

    def _compute_circumcircle(self):
        """Compute circumcircle parameters from backward-transform vertices."""
        ax, ay = self.A[2], self.A[3]
        bx, by = self.B[2], self.B[3]
        cx, cy = self.C[2], self.C[3]
        a2 = distsquare(bx, by, cx, cy)
        b2 = distsquare(ax, ay, cx, cy)
        c2 = distsquare(ax, ay, bx, by)
        fa = a2 * (b2 + c2 - a2)
        fb = b2 * (c2 + a2 - b2)
        fc = c2 * (a2 + b2 - c2)
        self.den = fa + fb + fc
        self.Mdenx = fa * ax + fb * bx + fc * cx
        self.Mdeny = fa * ay + fb * by + fc * cy
        self.r2den = distsquare(ax * self.den, ay * self.den, self.Mdenx, self.Mdeny)

    def removeedges(self):
        """
        Removes the edges of the triangle.
        """
        for edge in self.edges:
            self.elist[edge] -= 1
        del self.edges
        del self.elist

    def incircle(self, x, y):
        """
        Checks if a point is inside the circumcircle of the triangle.

        Args:
            x (float): X coordinate of the point.
            y (float): Y coordinate of the point.

        Returns:
            bool: True if the point is inside the circumcircle, False otherwise.
        """
        return (
            distsquare(x * self.den, y * self.den, self.Mdenx, self.Mdeny) < self.r2den
        )

    def contains_point(self, x, y):
        """Return True if scalar point (x, y) lies inside this triangle."""
        ok, _, _ = self.intriangle_subset(
            np.asarray([x], dtype=np.float64),
            np.asarray([y], dtype=np.float64),
        )
        return bool(ok[0])

    def _barycentric_transform(self, x, y, decomp, xi, yi):
        """Return transformed coordinates for points that lie inside triangle."""
        u = x * decomp[0, 0] + y * decomp[1, 0] + decomp[2, 0]
        v = x * decomp[0, 1] + y * decomp[1, 1] + decomp[2, 1]

        ok = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (u + v <= 1)
        if not np.any(ok):
            return ok, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        u_ok = u[ok]
        v_ok = v[ok]
        x_prime = (
            self.A[xi]
            + (self.B[xi] - self.A[xi]) * u_ok
            + (self.C[xi] - self.A[xi]) * v_ok
        )
        y_prime = (
            self.A[yi]
            + (self.B[yi] - self.A[yi]) * u_ok
            + (self.C[yi] - self.A[yi]) * v_ok
        )
        return ok, x_prime, y_prime

    def intriangle_subset(self, x, y):
        """Transform subset points that lie inside this triangle (backward map)."""
        return self._barycentric_transform(x, y, self.decomp, 0, 1)

    def inforward_subset(self, x, y):
        """Transform subset points that lie inside this triangle (forward map)."""
        return self._barycentric_transform(x, y, self.forwarddecomp, 2, 3)
