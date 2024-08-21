"""This code was written by Gergely Csucs and Rembrandt Bakker"""
import numpy as np


def triangulate(w, h, markers):
    vertices = [
        [-0.1 * w, -0.1 * h, -0.1 * w, -0.1 * h],
        [1.1 * w, -0.1 * h, 1.1 * w, -0.1 * h],
        [-0.1 * w, 1.1 * h, -0.1 * w, 1.1 * h],
        [1.1 * w, 1.1 * h, 1.1 * w, 1.1 * h],
    ]

    edges = [0] * ((len(markers) + 4) * (len(markers) + 4 - 1) // 2)
    triangles = [Triangle(0, 1, 2, vertices, edges), Triangle(1, 2, 3, vertices, edges)]
    edges[0] = edges[1] = edges[4] = edges[5] = 2
    markers = list(set(tuple(m) for m in markers))
    markers = [list(m) for m in markers]

    for marker in markers:
        x, y = marker[2:4]
        found = False
        keep = []
        remove = []
        for triangle in triangles:
            if not found and triangle.intriangle(x, y):
                found = True
            if triangle.incircle(x, y):
                remove.append(triangle)
            else:
                keep.append(triangle)
        if found:
            for triangle in remove:
                triangle.removeedges()
        else:
            keep.extend(remove)
        triangles = keep
        vcount = len(vertices)
        vertices.append(marker)
        for i in range(vcount - 1):
            for j in range(i + 1, vcount):
                if edges[edgeindex(i, j)] == 1:
                    triangles.append(Triangle(i, j, vcount, vertices, edges))
    return triangles


def transform(triangulation, x, y):
    for triangle in triangulation:
        uv1 = triangle.intriangle(x, y)
        if uv1:
            return (
                triangle.A[0]
                + (triangle.B[0] - triangle.A[0]) * uv1[0]
                + (triangle.C[0] - triangle.A[0]) * uv1[1],
                triangle.A[1]
                + (triangle.B[1] - triangle.A[1]) * uv1[0]
                + (triangle.C[1] - triangle.A[1]) * uv1[1],
            )


def transform_vec(triangulation, x, y):
    xPrime = np.zeros(x.shape, float)
    yPrime = np.zeros(y.shape, float)
    for triangle in triangulation:
        triangle.intriangle_vec(x, y, xPrime, yPrime)
    return (xPrime, yPrime)


def forwardtransform(triangulation, x, y):
    for triangle in triangulation:
        uv1 = triangle.inforward(x, y)
        if uv1:
            return (
                triangle.A[2]
                + (triangle.B[2] - triangle.A[2]) * uv1[0]
                + (triangle.C[2] - triangle.A[2]) * uv1[1],
                triangle.A[3]
                + (triangle.B[3] - triangle.A[3]) * uv1[0]
                + (triangle.C[3] - triangle.A[3]) * uv1[1],
            )


# xy: 2-dimensional array with one xy-pair per row
def forwardtransform_vec(triangulation, x, y):
    xPrime = np.zeros(x.shape, float)
    yPrime = np.zeros(y.shape, float)
    for triangle in triangulation:
        triangle.inforward_vec(x, y, xPrime, yPrime)
    return (xPrime, yPrime)


def inv3x3(m):
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if det == 0:
        return None
    return [
        [
            (m[1][1] * m[2][2] - m[2][1] * m[1][2]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[1][0] * m[0][2] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[2][0] * m[1][1]) / det,
            (m[2][0] * m[0][1] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[1][0] * m[0][1]) / det,
        ],
    ]


def rowmul3(v, m):
    return [sum(v[j] * m[j][i] for j in range(3)) for i in range(3)]


def rowmul3_vec(x, y, m):
    return np.outer(x, m[0]) + np.outer(y, m[1]) + m[2]


def distsquare(ax, ay, bx, by):
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by)


def edgeindex(a, b):
    i = min(a, b)
    j = max(a, b)
    return j * (j - 1) // 2 + i


class Triangle:
    def __init__(self, a, b, c, vlist, elist):
        self.A = vlist[a]
        self.B = vlist[b]
        self.C = vlist[c]
        self.elist = elist
        self.edges = [edgeindex(a, b), edgeindex(a, c), edgeindex(b, c)]
        for edge in self.edges:
            elist[edge] += 1
        ax, ay = self.A[0:2]
        bx, by = self.B[0:2]
        cx, cy = self.C[0:2]
        self.forwarddecomp = inv3x3(
            [[bx - ax, by - ay, 0], [cx - ax, cy - ay, 0], [ax, ay, 1]]
        )
        ax, ay = self.A[2:4]
        bx, by = self.B[2:4]
        cx, cy = self.C[2:4]
        self.decomp = inv3x3(
            [[bx - ax, by - ay, 0], [cx - ax, cy - ay, 0], [ax, ay, 1]]
        )
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
        for edge in self.edges:
            self.elist[edge] -= 1
        del self.edges
        del self.elist

    def incircle(self, x, y):
        return (
            distsquare(x * self.den, y * self.den, self.Mdenx, self.Mdeny) < self.r2den
        )

    def intriangle(self, x, y):
        uv1 = rowmul3([x, y, 1], self.decomp)
        if 0 <= uv1[0] <= 1 and 0 <= uv1[1] <= 1 and uv1[0] + uv1[1] <= 1:
            return uv1

    def inforward(self, x, y):
        uv1 = rowmul3([x, y, 1], self.forwarddecomp)
        if 0 <= uv1[0] <= 1 and 0 <= uv1[1] <= 1 and uv1[0] + uv1[1] <= 1:
            return uv1

    # xy: 2-dimensional array with one xy-pair per row
    def inforward_vec(self, x, y, xPrime, yPrime):
        uv1 = rowmul3_vec(x, y, self.forwarddecomp)
        # also compute the next step, since it uses the parameters of this triangle
        ok = (
            (uv1[:, 0] >= 0)
            & (uv1[:, 0] <= 1)
            & (uv1[:, 1] >= 0)
            & (uv1[:, 1] <= 1)
            & (uv1[:, 0] + uv1[:, 1] <= 1)
        )
        xPrime[ok] = (
            self.A[2]
            + (self.B[2] - self.A[2]) * uv1[ok, 0]
            + (self.C[2] - self.A[2]) * uv1[ok, 1]
        )
        yPrime[ok] = (
            self.A[3]
            + (self.B[3] - self.A[3]) * uv1[ok, 0]
            + (self.C[3] - self.A[3]) * uv1[ok, 1]
        )

    def intriangle_vec(self, x, y, xPrime, yPrime):
        uv1 = rowmul3_vec(x, y, self.decomp)
        # also compute the next step, since it uses the parameters of this triangle
        ok = (
            (uv1[:, 0] >= 0)
            & (uv1[:, 0] <= 1)
            & (uv1[:, 1] >= 0)
            & (uv1[:, 1] <= 1)
            & (uv1[:, 0] + uv1[:, 1] <= 1)
        )
        xPrime[ok] = (
            self.A[0]
            + (self.B[0] - self.A[0]) * uv1[ok, 0]
            + (self.C[0] - self.A[0]) * uv1[ok, 1]
        )
        yPrime[ok] = (
            self.A[1]
            + (self.B[1] - self.A[1]) * uv1[ok, 0]
            + (self.C[1] - self.A[1]) * uv1[ok, 1]
        )
