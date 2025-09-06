import numpy as np
import open3d as o3d
import torch


def R_from_two_direction(origin_vector, location_vector):
    c = np.dot(origin_vector, location_vector)
    n_vector = np.cross(origin_vector, location_vector)
    s = np.linalg.norm(n_vector)

    n_vector_invert = np.array((
        [0, -n_vector[2], n_vector[1]],
        [n_vector[2], 0, -n_vector[0]],
        [-n_vector[1], n_vector[0], 0]
    ))
    I = np.eye(3)
    # 核心公式：见上图
    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
    return R_w2c


def line(st_pt, ed_pt, width1=0.003, width2=0.003):
    line_dir = ed_pt - st_pt
    line_dir = line_dir / np.linalg.norm(line_dir, axis=0).item()
    R = R_from_two_direction(np.array([0, 0, 1]), line_dir)
    unit_ring1 = np.array([
        [1, 0, 0],
        # [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        [0, 1, 0],
        # [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        [-1, 0, 0],
        # [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
        [0, -1, 0],
        # [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
    ])*width1
    unit_ring2 = np.array([
        [1, 0, 0],
        # [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        [0, 1, 0],
        # [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        [-1, 0, 0],
        # [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
        [0, -1, 0],
        # [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
    ]) * width2
    unit_ring1 = unit_ring1.dot(R.T)
    unit_ring2 = unit_ring2.dot(R.T)
    circle_st = unit_ring1 + st_pt[None, :]
    circle_ed = unit_ring2 + ed_pt[None, :]

    # vec = np.zeros((2, unit_ring.shape[0], 3))
    # for h_k in range(vec.shape[0]):
    #     low_h = h_cross[0] + h_k * h_patch
    #     vec[h_k, :, :] = np.concatenate([l_contour[h_k], np.array([[low_h]] * l_contour.shape[1])], axis=1)
    vec = np.stack([circle_st, circle_ed], axis=0)

    tris = []
    for h_k in range(1, vec.shape[0]):
        fa_id, cur_id = h_k - 1, h_k
        for j in range(vec.shape[1]):
            # v1 = cur_ring[j]
            v1 = cur_id * vec.shape[1] + j
            # v2 = cur_ring[(j + 1) % 8]
            v2 = cur_id * vec.shape[1] + (j + 1) % vec.shape[1]
            # v3 = fa_ring[j]
            v3 = fa_id * vec.shape[1] + j
            # v4 = fa_ring[(j + 1) % 8]
            v4 = fa_id * vec.shape[1] + (j + 1) % vec.shape[1]
            tris.append([v3, v2, v1])
            tris.append([v2, v3, v4])
    vec = vec.reshape(-1, 3)
    # 封口
    vec = np.concatenate([vec, st_pt[None, :], ed_pt[None, :]], axis=0)
    st_vec_ind = 2 * unit_ring1.shape[0]
    ed_vec_ind = st_vec_ind + 1
    for i in range(2):
        cur_ring = [circle_st, circle_ed][i]
        v3 = [st_vec_ind, ed_vec_ind][i]
        for j in range(cur_ring.shape[0]):
            v1 = i * cur_ring.shape[0] + j
            # v2 = cur_ring[(j + 1) % 8]
            v2 = i * cur_ring.shape[0] + (j + 1) % cur_ring.shape[0]
            if i == 0:
                tris.append([v3, v2, v1])
            else:
                tris.append([v1, v2, v3])

    tris = np.array(tris)
    line_mesh = o3d.geometry.TriangleMesh()
    line_mesh.vertices = o3d.utility.Vector3dVector(vec)
    line_mesh.triangles = o3d.utility.Vector3iVector(tris)
    line_mesh.compute_vertex_normals()
    return line_mesh
