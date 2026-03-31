import numpy as np


class Vector:
    """Wraps a 3D point/vector with common geometric operations."""

    def __init__(self, array):
        self.array = list(array)

    def norm(self):
        return np.sqrt(sum(x ** 2 for x in self.array))

    def unit_vec(self):
        n = self.norm()
        if n == 0:
            return [0.0, 0.0, 0.0]
        return [x / n for x in self.array]

    def multi(self, scalar):
        return Vector([x * scalar for x in self.array])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.array, other.array))


class Line:
    """Line through the origin in the xz-plane with slope grad (z = grad * x)."""

    def __init__(self, grad):
        self.grad = grad

    def get_x(self, z):
        """Return x coordinate for a given z value: x = z / grad."""
        return z / self.grad


class Face:
    """Quadrilateral face defined by 4 Vector vertices (in order).

    Vertices should be specified in counter-clockwise order when viewed from
    outside the polyhedron so that the cross product gives an outward normal.
    Uses vertices 0, 1 and 3 to compute the edge vectors for the normal.
    """

    def __init__(self, vertices):
        self.vertices = vertices  # list of 4 Vector objects
        v0 = np.array(vertices[0].array, dtype=float)
        v1 = np.array(vertices[1].array, dtype=float)
        v3 = np.array(vertices[3].array, dtype=float)
        e1 = v1 - v0
        e2 = v3 - v0
        self.normal = np.cross(e1, e2)  # outward-pointing normal
        self.point = v0                 # any point on the face

    def is_inside(self, pt):
        """Return True if pt is on the interior side of this face."""
        pt_arr = np.array(pt, dtype=float)
        return float(np.dot(self.normal, pt_arr - self.point)) <= 0.0


def isInPoly(pt, faces):
    """Return True if pt is inside the convex polyhedron defined by faces.

    pt   -- raw 3-element list/array
    faces -- list of Face objects whose normals point outward
    """
    for face in faces:
        if not face.is_inside(pt):
            return False
    return True
