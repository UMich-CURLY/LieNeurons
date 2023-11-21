import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import matplotlib.pyplot as plt

from scipy.linalg import logm
from core.lie_alg_util import *


def construct_platonic_solids():
    platonic_solids = dict()

    # Tetrahedron
    tetrahedron_vertices = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ], dtype=np.float32)

    tetrahedron_faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2]
    ], dtype=np.uint32)

    platonic_solids['tetra'] = {
        'vertices': tetrahedron_vertices, 'faces': tetrahedron_faces}

    # Cube
    cube_vertices = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1]
    ], dtype=np.float32)

    cube_faces = np.array([
        [0, 1, 2, 3],
        [3, 2, 6, 7],
        [7, 6, 5, 4],
        [4, 5, 1, 0],
        [1, 5, 6, 2],
        [4, 0, 3, 7]
    ], dtype=np.uint32)

    platonic_solids['cube'] = {'vertices': cube_vertices, 'faces': cube_faces}

    # Octahedron
    octahedron_vertices = np.array([
        [0, 0, 1],
        [0, 0, -1],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ], dtype=np.float32)

    octahedron_faces = np.array([
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5]
    ], dtype=np.uint32)

    platonic_solids['octa'] = {
        'vertices': octahedron_vertices, 'faces': octahedron_faces}

    # Dodecahedron
    dodecahedron_vertices = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
        [0, (1 + np.sqrt(5)) / 2, 1 / 2],
        [0, (1 + np.sqrt(5)) / 2, -1 / 2],
        [0, -(1 + np.sqrt(5)) / 2, 1 / 2],
        [0, -(1 + np.sqrt(5)) / 2, -1 / 2],
        [(1 + np.sqrt(5)) / 2, 1 / 2, 0],
        [(1 + np.sqrt(5)) / 2, -1 / 2, 0],
        [-(1 + np.sqrt(5)) / 2, 1 / 2, 0],
        [-(1 + np.sqrt(5)) / 2, -1 / 2, 0],
        [1 / 2, 0, (1 + np.sqrt(5)) / 2],
        [-1 / 2, 0, (1 + np.sqrt(5)) / 2],
        [1 / 2, 0, -(1 + np.sqrt(5)) / 2],
        [-1 / 2, 0, -(1 + np.sqrt(5)) / 2]
    ], dtype=np.float32)

    dodecahedron_faces = np.array([
        [0, 12, 4, 14, 8],
        [0, 8, 10, 2, 16],
        [0, 16, 6, 18, 12],
        [1, 9, 11, 3, 17],
        [1, 17, 7, 19, 13],
        [1, 13, 5, 15, 9],
        [2, 10, 14, 4, 18],
        [2, 18, 6, 16, 8],
        [3, 11, 19, 7, 17],
        [3, 17, 5, 13, 9],
        [3, 9, 15, 7, 19],
        [4, 12, 18, 6, 14]
    ], dtype=np.uint32)

    platonic_solids['dodeca'] = {
        'vertices': dodecahedron_vertices, 'faces': dodecahedron_faces}

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    icosahedron_vertices = np.array([
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1]
    ], dtype=np.float32)

    icosahedron_faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ], dtype=np.uint32)

    platonic_solids['icosa'] = {
        'vertices': icosahedron_vertices, 'faces': icosahedron_faces}

    return platonic_solids

# Function to compute neighboring faces for each face


def compute_neighbors(platonic_solid):
    faces = platonic_solid['faces']
    neighbors = {}

    for i, face in enumerate(faces):
        neighbors[i] = []
        for j, other_face in enumerate(faces):
            if i != j:
                common_vertices = np.intersect1d(face, other_face)
                if len(common_vertices) == 2:
                    neighbors[i].append(j)

    return neighbors


def compute_neighboring_faces_for_each_vertice(platonic_solid):
    faces = platonic_solid['faces']
    vertices = platonic_solid['vertices']
    neighboring_faces_for_each_vertice = {}

    for i, vertice in enumerate(vertices):
        neighboring_faces_for_each_vertice[i] = []
        for j, face in enumerate(faces):
            if i in face:
                neighboring_faces_for_each_vertice[i].append(j)

    return neighboring_faces_for_each_vertice


def compute_neighboring_vertice_for_each_vertice(platonic_solid):

    # Go through each face, for each vertice in the face, it is connected to the next vertice in the face.
    # The last vertice in the face is connected to the first vertice in the face.

    neighboring_vertice_for_each_vertice = {}

    for face in platonic_solid['faces']:
        for i, vertice in enumerate(face):
            if vertice not in neighboring_vertice_for_each_vertice:
                neighboring_vertice_for_each_vertice[vertice] = []
            neighbor_idx = (i+1) % len(face)
            face_neighbor = face[neighbor_idx]
            if face_neighbor not in neighboring_vertice_for_each_vertice[vertice]:
                neighboring_vertice_for_each_vertice[vertice].append(
                    face_neighbor)

    # print(neighboring_vertice_for_each_vertice)
    return neighboring_vertice_for_each_vertice


def compute_neighboring_faces_for_each_ordered_edge(platonic_solid):
    # Each edge is represented by a tuple of two vertices (order matters)
    # We want to compute the neighboring faces for each edge
    # The order of two neighboring faces also matters.
    # Viewing from the first vertice the edge to the second,
    # the first face is the one on the left, the second face is the one on the right.

    neighboring_faces_for_each_ordered_edge = {}

    neighboring_vertice_for_each_vertice = compute_neighboring_vertice_for_each_vertice(
        platonic_solid)

    # Calculate the center of the solid
    vertices = platonic_solid['vertices']
    center = np.mean(vertices, axis=0)

    for vertice in neighboring_vertice_for_each_vertice:
        for neighbor in neighboring_vertice_for_each_vertice[vertice]:
            edge = (vertice, neighbor)
            neighboring_faces_for_each_ordered_edge[edge] = dict()
            for face in platonic_solid['faces']:
                if vertice in face and neighbor in face:
                    # neighboring_faces_for_each_ordered_edge[edge].append(face)

                    vertice3 = np.setdiff1d(face, edge)[0]
                    edge1 = vertices[neighbor] - vertices[vertice]
                    edge2 = vertices[vertice3] - vertices[vertice]
                    cross_product = np.cross(edge1, edge2)
                    # vector from solid center to the first vertice in the edge
                    vector = (
                        vertices[vertice] + vertices[neighbor] + vertices[vertice3])/3 - center
                    # print(vector)
                    if np.dot(cross_product, vector) > 0:
                        neighboring_faces_for_each_ordered_edge[edge]['left'] = face
                    else:
                        neighboring_faces_for_each_ordered_edge[edge]['right'] = face

    # ### Now we need to order the neighboring faces for each edge
    # ### We do this by computing the cross product of the shared edge
    # ### and the edge connecting the first vertice in the shared edge and the other vertice in the face
    # ### The direction of the cross product tells us the order of the two neighboring faces
    # ### If the cross product points outwards, the face is on the left. Otherwise, it is on the right.

    return neighboring_faces_for_each_ordered_edge

# For each ordered pair of faces neighboring an ordered edge, calculate the homography matrix
# that maps the first face to the second face


def calculate_homography_dlt(pfrom, pto):
    p1, p2, p3, p4 = pfrom
    p1_prime, p2_prime, p3_prime, p4_prime = pto

    A = np.zeros((8, 9))
    A[0, 0:3] = p1
    A[0, 6:9] = -p1*p1_prime[0]
    A[1, 3:6] = p1
    A[1, 6:9] = -p1*p1_prime[1]
    A[2, 0:3] = p2
    A[2, 6:9] = -p2*p2_prime[0]
    A[3, 3:6] = p2
    A[3, 6:9] = -p2*p2_prime[1]
    A[4, 0:3] = p3
    A[4, 6:9] = -p3*p3_prime[0]
    A[5, 3:6] = p3
    A[5, 6:9] = -p3*p3_prime[1]
    A[6, 0:3] = p4
    A[6, 6:9] = -p4*p4_prime[0]
    A[7, 3:6] = p4
    A[7, 6:9] = -p4*p4_prime[1]

    U, S, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)

    return H


def compute_homography_matrix_for_each_ordered_edge(platonic_solid):
    neighboring_faces_for_each_ordered_edge = compute_neighboring_faces_for_each_ordered_edge(
        platonic_solid)
    homography_matrix_for_each_ordered_edge = {}
    projected_points = {}

    for edge in neighboring_faces_for_each_ordered_edge:
        face1 = neighboring_faces_for_each_ordered_edge[edge]['left']
        face2 = neighboring_faces_for_each_ordered_edge[edge]['right']
        vertice1 = edge[0]
        vertice2 = edge[1]
        vertice3 = np.setdiff1d(face1, edge)[0]
        vertice4 = np.setdiff1d(face2, edge)[0]

        # point coordinates
        p1 = platonic_solid['vertices'][vertice1]
        p2 = platonic_solid['vertices'][vertice2]
        p3 = platonic_solid['vertices'][vertice3]
        p4 = platonic_solid['vertices'][vertice4]

        # compute homography matrix from the plane containing face1 to the plane containing face2
        # the two planes now only have three points. To solve for homography matrix, we need at least four points.
        # We construct the fourth point as the sum of the shared edge
        # and the edge connecting the first vertice in the shared edge and the other vertice in the face

        p5 = p2 + p3 - p1
        p6 = p2 + p4 - p1
        p7 = p4 + p1 - p2
        p8 = p1 + p2 - p4

        # projection of points to the unit depth plane
        p1 = p1/p1[2]

        p2 = p2/p2[2]
        p3 = p3/p3[2]
        p4 = p4/p4[2]
        p5 = p5/p5[2]
        p6 = p6/p6[2]
        p7 = p7/p7[2]
        p8 = p8/p8[2]
        # p1, p2, p3, p4, p5, p6 are projections of points to the unit depth plane
        projected_points[edge] = np.array([p1, p2, p3, p4, p5, p6, p7, p8])

        # Use DLT algorithm to solve for homography matrix
        # solve for homography matrix mapping p1, p2, p5, p3 to p1, p4, p6, p2
        H_1 = calculate_homography_dlt([p1, p2, p5, p3], [p1, p4, p6, p2])

        # solve for homography matrix mapping p1, p2, p5, p3 to p2, p1, p7, p4
        H_2 = calculate_homography_dlt([p1, p2, p5, p3], [p2, p1, p7, p4])

        # solve for homography matrix mapping p1, p2, p5, p3 to p4, p2, p8, p1
        H_3 = calculate_homography_dlt([p1, p2, p5, p3], [p4, p2, p8, p1])

        homography_matrix_for_each_ordered_edge[edge] = np.array([
                                                                 H_1, H_2, H_3])

    return homography_matrix_for_each_ordered_edge, projected_points


def compute_sl3_from_homography_matrices(homographis):
    N, F, _, _ = np.shape(homographis)
    sl3 = np.zeros((N, F, 3, 3))
    for n in range(N):
        for f in range(F):
            det = np.linalg.det(homographis[n, f, :, :])
            if det < 0:
                scale = -(-det)**(-1/3)
            elif det > 0:
                scale = det**(-1/3)
            else:
                scale = 1e-6

            sl3[n, f, :, :] = logm(homographis[n, f, :, :]*scale)
    return sl3

# Specify the location of the platonic solid and the camera.


class PlatonicDataset(Dataset):
    def __init__(self, data_set_size) -> None:
        super().__init__()
        self.len = data_set_size
        self.n_sample_per_item = 12

        self.platonic_solids = construct_platonic_solids()
        # octa = self.platonic_solids['octa']     # 8 faces
        # icosa = self.platonic_solids['icosa']   # 20 faces
        # tetra = self.platonic_solids['tetra']   # 4 faces

        center = np.array([[5, 3, 10]], dtype=np.float32)
        for platonic_solid in self.platonic_solids:
            self.platonic_solids[platonic_solid]['vertices'] = self.platonic_solids[platonic_solid]['vertices'] + center

        self.homographis = dict()
        self.sl3 = dict()
        self.projected_points = dict()
        for platonic_solid in ['octa', 'icosa', 'tetra']:  # 'octa', 'icosa', 'tetra'
            self.homographis[platonic_solid], self.projected_points[platonic_solid] = compute_homography_matrix_for_each_ordered_edge(
                self.platonic_solids[platonic_solid])
            self.sl3[platonic_solid] = compute_sl3_from_homography_matrices(
                np.array(list(self.homographis[platonic_solid].values())))

        # self.visualize('octa')

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        # cls = np.random.choice(['octa', 'icosa', 'tetra'])
        cls = np.random.choice(['octa', 'icosa', 'tetra'])

        # cls = 'tetra'
        platonic_solid = self.platonic_solids[cls]
        # homographies = list(self.homographis[cls].values())

        # homographies = np.array(homographies)
        sl3 = self.sl3[cls]

        # print(homographies)   # (12(n), 3(number of h), 3, 3)
        # sample self.n_sample_per_item homographies from the homographies of the platonic solid
        # each homography is a list of three homography matrices
        # each homography matrix maps one face to another face
        # the order of the two faces matters.

        sl3_idx = np.random.choice(
            list(range(sl3.shape[0])), size=self.n_sample_per_item)
        sl3_sampled = sl3[sl3_idx]
        # print(sl3_sampled.shape)   # (12(n), F=3(number of h), 3, 3)
        cls_int = {'octa': 0, 'icosa': 1, 'tetra': 2}[cls]
        sl3_sampled_vee = vee_sl3(torch.from_numpy(
            sl3_sampled).type('torch.FloatTensor'))
        return sl3_sampled_vee, torch.tensor(cls_int, dtype=torch.int8).type(torch.LongTensor)

    def visualize(self, shape='tetra'):
        octa = self.platonic_solids[shape]

        proj_pts = list(self.projected_points[shape].values())
        proj_pts = np.array(proj_pts)

        # visualize the projected octahedron vertices and edges on the unit-depth plane
        # print(proj_pts.shape)   # n_pairs, 8, 3
        proj_pts_vertices = proj_pts[:, :4, :]
        proj_pts_vertices = proj_pts_vertices.reshape(-1, 3)
        plt.plot(proj_pts_vertices[:, 0], proj_pts_vertices[:, 1], 'r.')
        plt.gca().set_aspect('equal')
        # plt.show()

        edges = list(self.projected_points[shape].keys())
        # draw the edges
        for edge in edges:
            p0 = self.projected_points[shape][edge][0][:2]
            p1 = self.projected_points[shape][edge][1][:2]
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b')

        # specify a set of points and warp using the homography
        homos = self.homographis[shape]

        # take edge 0 as example
        edge0 = edges[0]
        pts_from = self.projected_points[shape][edge0][:3].T
        pts_to = self.projected_points[shape][edge0][[0, 3, 1]].T
        homo = self.homographis[shape][edge0][0]
        pts_homo = homo @ pts_from
        pts_homo = pts_homo / pts_homo[[2]]

        plt.plot(pts_from[0, 0], pts_from[1, 0], 'ro',
                 markersize=20, markerfacecolor='none')
        plt.plot(pts_from[0, 1], pts_from[1, 1], 'go',
                 markersize=20, markerfacecolor='none')
        plt.plot(pts_from[0, 2], pts_from[1, 2], 'bo',
                 markersize=20, markerfacecolor='none')

        plt.plot(pts_to[0, 0], pts_to[1, 0], 'rx',
                 markersize=20, markerfacecolor='none')
        plt.plot(pts_to[0, 1], pts_to[1, 1], 'gx',
                 markersize=20, markerfacecolor='none')
        plt.plot(pts_to[0, 2], pts_to[1, 2], 'bx',
                 markersize=20, markerfacecolor='none')

        plt.plot(pts_homo[0, 0], pts_homo[1, 0], 'rD',
                 markersize=20, markerfacecolor='none')
        plt.plot(pts_homo[0, 1], pts_homo[1, 1], 'gD',
                 markersize=20, markerfacecolor='none')
        plt.plot(pts_homo[0, 2], pts_homo[1, 2], 'bD',
                 markersize=20, markerfacecolor='none')


if __name__ == '__main__':
    platonic_dataset = PlatonicDataset(device='cpu')
    platonic_dataset.__getitem__(0)

    plt.show()
