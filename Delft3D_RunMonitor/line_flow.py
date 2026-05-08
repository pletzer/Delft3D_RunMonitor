import numpy as np

class LineFlow:

    def __init__(self, ugrid, xa, ya, xb, yb):
        """
        Constructor
        :param ugrid: UGRID ugrid_mesh instance
        :param xa: start x coorindate
        :param ya: start y coorindate
        :param xb: end x coordinate
        :param yb: end y coordinate
        """
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb

        # 1D array of x coordinates
        self.x = ugrid.x
        # 1D array of y coordinates
        self.y = ugrid.y

        # 2D array of 3 coordinate ids for each face
        self.face_nodes = ugrid.face_nodes

        # 2D array of 2 coordinate ids for each edge
        self.edge_nodes = ugrid.edge_nodes

        # self.face_edge = {}
        # # TO IMPLEMENT. This is a map of {face_id: [(edge_id0, sign0), (edge_id1, sign1), (edge_id2, sign2)], ...}

        # Build edge lookup
        edge_lookup = {}

        for edge_id, (n0, n1) in enumerate(self.edge_nodes):
            edge_lookup[(n0, n1)] = (edge_id, +1)
            edge_lookup[(n1, n0)] = (edge_id, -1)

        # face_id -> [(edge_id, sign), ...]
        for face_id, face in enumerate(self.face_nodes):

            n0, n1, n2 = face

            self.face_edge[face_id] = [
                edge_lookup[(n0, n1)],   # local edge 01
                edge_lookup[(n1, n2)],   # local edge 12
                edge_lookup[(n2, n0)],   # local edge 20
            ]

        self._collect_crossed_faces()

    def _collect_crossed_faces(self):
        """
        Find all the faces intersected by the (xa, ya) -> (xb, yb) line
        and save the result as {face_id: (xi_a, eta_a, xi_b, eta_b)} where 
        0 <= xi_a, eta_a <= 1 are the start and 0 <= xi_b, eta_b <= 1 are the
        end parametric coordinates in the triangle face_id. Include the start/end
        points if these fall inside a triangular face. 
        """

        self.face_param_coords = {}

        pa = np.array([self.xa, self.ya])
        pb = np.array([self.xb, self.yb])

        d = pb - pa
        seg_len2 = np.dot(d, d)

        tol = 1.0e-12

        for face_id, nodes in enumerate(self.face_nodes):

            verts = np.array([
                [self.x[nodes[0]], self.y[nodes[0]]],
                [self.x[nodes[1]], self.y[nodes[1]]],
                [self.x[nodes[2]], self.y[nodes[2]]],
            ])

            # ---------------------------------------------------------
            # Compute barycentric coordinates of endpoints
            # ---------------------------------------------------------
            def barycentric(p):

                a = verts[0]
                b = verts[1]
                c = verts[2]

                v0 = b - a
                v1 = c - a
                v2 = p - a

                det = v0[0]*v1[1] - v1[0]*v0[1]

                if abs(det) < tol:
                    return None

                xi = (v2[0]*v1[1] - v1[0]*v2[1]) / det
                eta = (v0[0]*v2[1] - v2[0]*v0[1]) / det

                return np.array([xi, eta])

            # ---------------------------------------------------------
            # Intersect line with triangle edges
            # ---------------------------------------------------------
            intersections = []

            tri_edges = [(0,1), (1,2), (2,0)]

            for i0, i1 in tri_edges:

                p0 = verts[i0]
                p1 = verts[i1]

                A = np.array([
                    [d[0], -(p1-p0)[0]],
                    [d[1], -(p1-p0)[1]],
                ])

                rhs = p0 - pa

                det = np.linalg.det(A)

                if abs(det) < tol:
                    continue

                t, s = np.linalg.solve(A, rhs)

                if (-tol <= t <= 1.0+tol and
                    -tol <= s <= 1.0+tol):

                    intersections.append((t, pa + t*d))

            # ---------------------------------------------------------
            # Include endpoints inside triangle
            # ---------------------------------------------------------
            bary_a = barycentric(pa)
            bary_b = barycentric(pb)

            inside_a = (
                bary_a is not None and
                bary_a[0] >= -tol and
                bary_a[1] >= -tol and
                bary_a.sum() <= 1.0 + tol
            )

            inside_b = (
                bary_b is not None and
                bary_b[0] >= -tol and
                bary_b[1] >= -tol and
                bary_b.sum() <= 1.0 + tol
            )

            if inside_a:
                intersections.append((0.0, pa))

            if inside_b:
                intersections.append((1.0, pb))

            # ---------------------------------------------------------
            # Remove duplicate intersections
            # ---------------------------------------------------------
            if len(intersections) < 2:
                continue

            intersections.sort(key=lambda x: x[0])

            unique = [intersections[0]]

            for item in intersections[1:]:
                if abs(item[0] - unique[-1][0]) > tol:
                    unique.append(item)

            if len(unique) < 2:
                continue

            # first/last point define clipped segment
            ta, qa = unique[0]
            tb, qb = unique[-1]

            bary_qa = barycentric(qa)
            bary_qb = barycentric(qb)

            xi_a, eta_a = bary_qa
            xi_b, eta_b = bary_qb

            self.face_param_coords[face_id] = (
                xi_a, eta_a,
                xi_b, eta_b
            )


    def _computeWeights(self):
        self.weights = {}
        for face_id, param_coords in self.face_param_coords.items():
            # compute the Whitneys for the 3 edges, w01, w12 and w20
            # assuming the nodes are ordered anticlockwise.
            xi_a, eta_a, xi_b, eta_b = param_coords
            xi_mean = 0.5*(xi_a + xi_b)
            eta_mean = 0.5*(eta_a + eta_b)
            dxi = xi_b - xi_a
            deta = eta_b - eta_a
            # Whitneys
            w01 = dxi * (1.0 - xi_mean - eta_mean) - (-dxi - deta) * xi_mean
            w12 = deta * xi_mean - dxi * eta_mean
            w20 = -(dxi + deta) * eta_mean - deta * (1.0 - xi_mean - eta_mean)

            # first edge 0 -> 1
            edge_id, sign = self.face_edge[face_id][0]
            self.weights[edge_id] = self.weights.get(edge_id, 0.0) + w01 * sign

            # second edge 1 -> 2
            edge_id, sign = self.face_edge[face_id][1]
            self.weights[edge_id] = self.weights.get(edge_id, 0.0) + w12 * sign

            # third edge 2 -> 0
            edge_id, sign = self.face_edge[face_id][2]
            self.weights[edge_id] = self.weights.get(edge_id, 0.0) + w20 * sign




    def getFlux(self, uData):
        """
        Compute the flux across the line
        :param uData the vertical face integrated flux on each triangle edge (need to multiply u1 with edge length * depth)
        :returns flux
        """
        flux = 0.0
        for edge_id, weight in self.weights.items():
            flux += uData[edge_id] * weight
        return flux

