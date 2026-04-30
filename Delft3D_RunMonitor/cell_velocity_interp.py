import numpy as np


def triangle_area(vertices):
    """Return the signed area of a triangle.

    Parameters
    ----------
    vertices : (3, 2) array_like
        Triangle vertices ordered counterclockwise.
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    return 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


class RT0TriangleInterpolator:
    """Lowest-order Raviart-Thomas interpolator on a triangle.

    The degrees of freedom are the outward normal fluxes across the
    three edges:

        F_i = \int_{e_i} u · n_i ds

    where edge i is opposite vertex i.
    """

    def __init__(self, vertices, normal_velocities):
        """Construct the RT0 interpolant.

        Parameters
        ----------
        vertices : (3, 2) array_like
            Triangle vertices in counterclockwise order.
        normal_velocities : (3,) array_like
            Edge-normal velocities. Entry i corresponds to the edge
            opposite vertex i, with the outward normal orientation.
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.normal_velocities = np.asarray(normal_velocities, dtype=float)

        if self.vertices.shape != (3, 2):
            raise ValueError("vertices must have shape (3, 2)")
        if self.normal_velocities.shape != (3,):
            raise ValueError("normal_velocities must have shape (3,)")

        self.area = triangle_area(self.vertices)
        if self.area <= 0.0:
            raise ValueError("vertices must be ordered counterclockwise")

        self.edge_lengths = self._compute_edge_lengths()
        self.fluxes = self.normal_velocities * self.edge_lengths

    def _compute_edge_lengths(self):
        v = self.vertices
        return np.array([
            np.linalg.norm(v[2] - v[1]),  # edge opposite vertex 0
            np.linalg.norm(v[0] - v[2]),  # edge opposite vertex 1
            np.linalg.norm(v[1] - v[0]),  # edge opposite vertex 2
        ])

    def basis(self, points):
        """Evaluate the three RT0 basis functions.

        Parameters
        ----------
        points : (..., 2) array_like

        Returns
        -------
        phi : (..., 3, 2) ndarray
            phi[..., i, :] is basis function i.
        """
        p = np.asarray(points, dtype=float)
        scalar_input = p.ndim == 1
        p = np.atleast_2d(p)

        v = self.vertices
        denom = 2.0 * self.area

        phi = np.empty((p.shape[0], 3, 2), dtype=float)
        for i in range(3):
            phi[:, i, :] = (p - v[i]) / denom

        return phi[0] if scalar_input else phi

    def velocity(self, points):
        """Evaluate the interpolated velocity.

        Parameters
        ----------
        points : (2,) or (N, 2) array_like
            Evaluation point(s).

        Returns
        -------
        vel : (2,) or (N, 2) ndarray
            Interpolated velocity vector(s).
        """
        phi = self.basis(points)
        return np.tensordot(phi, self.fluxes, axes=([-2], [0]))

    def uv(self, points):
        """Return the interpolated u and v components separately."""
        vel = self.velocity(points)
        return vel[..., 0], vel[..., 1]


if __name__ == "__main__":
    # Example: reference triangle
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    # Edge-normal velocities (one per edge, opposite each vertex)
    un = np.array([0.5, -0.2, 0.1])

    interp = RT0TriangleInterpolator(vertices, un)

    x = np.array([0.2, 0.3])
    u, v = interp.uv(x)
    print(f"Velocity at {x}: u = {u:.6f}, v = {v:.6f}")

    pts = np.array([
        [0.2, 0.2],
        [0.4, 0.1],
        [0.2, 0.5],
    ])
    uv = interp.velocity(pts)
    print("\nVelocities:")
    print(uv)
