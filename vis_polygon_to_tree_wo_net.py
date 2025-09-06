import numpy as np
import torch
import open3d as o3d
from utils import get_cylinders, to_o3d_pcd, yellow
from models.sampler import DDIM
from vis_random_sample import lines_to_tree_structure, net
from datasets import renorm_w
from algorithm import uv_2
from copy import deepcopy
import cv2


def xyz_to_mesh(xyz, radius=0.008, resolution=20):
    pts, edges = [], []
    for i in range(xyz.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)

        np.asarray(sphere.vertices)[:] += xyz[i]

        pts.append(np.asarray(sphere.vertices))
        edges.append(np.asarray(sphere.triangles)+np.asarray(sphere.vertices).shape[0]*i)
    pts = np.concatenate(pts, axis=0)
    edges = np.concatenate(edges, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(edges)
    # mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector([yellow]*pts.shape[0])
    return mesh


def add_leaves_to_tree(tree_struct, pts, th=0.01):
    from utils import square_distance
    dis = square_distance(torch.Tensor(pts)[None, :, :].cuda(), torch.Tensor(tree_struct[None, :, :3]).cuda())[0]
    min_dis, min_inds = dis.min(dim=1)
    add_inds = (min_dis < th)
    is_leaf = torch.Tensor(tree_struct).cuda()[min_inds, 5].bool()
    add_inds = add_inds & is_leaf

    print("add num: %d" % (add_inds.sum(dim=0)))
    pts_corr_ind = min_inds[add_inds].cpu().numpy()
    pts = pts[add_inds.cpu().numpy()]
    # xyz, fa, width, is_leaf, leaf_size = data[:, :3], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
    pts_fa_w = tree_struct[pts_corr_ind, 4]
    add_data = np.concatenate([pts, pts_corr_ind[:, None], pts_fa_w[:, None], np.ones((pts.shape[0], 1)), tree_struct[pts_corr_ind, 6][:, None]], axis=1)
    new_tree_struct = np.copy(tree_struct)
    new_tree_struct = np.concatenate([new_tree_struct, add_data], axis=0)
    return new_tree_struct


if __name__ == '__main__':
    import time
    # 9  14  15  19  23  24  27  28
    poly = np.load("C:/Users/Administrator/Desktop/match_img_pcd/polygons/27.npy")
    print(poly)
    img = cv2.imread("C:/Users/Administrator/Desktop/match_img_pcd/img.jpg")
    img = cv2.polylines(img, [poly], isClosed=True, thickness=5,  color=(57, 0, 227))

    img = cv2.resize(img, (1024, 1024))
    cv2.imshow("", img)
    cv2.waitKey(0)

    xyz = np.concatenate([poly, -np.ones((poly.shape[0], 1))], axis=1)
    xyz_center = np.mean(xyz, axis=0, keepdims=True)
    xyz[:, :2] = xyz[:, :2] - xyz_center[:, :2]
    radius = np.linalg.norm(xyz[:, :2], axis=1).max().item()
    xyz[:, :2] = xyz[:, :2] / radius * 0.4
    xyz_ = np.copy(xyz)
    xyz[:, 2] += 1.8

    fa = [poly.shape[0]-1]+np.arange(poly.shape[0]-1).tolist()
    xyzf = np.concatenate([xyz, np.array(fa)[:, None]], axis=1)
    xyzf_ = np.concatenate([xyz_, np.array(fa)[:, None]], axis=1)
    cylinders = get_cylinders(xyzf, r=0.0025)
    cylinders_ = get_cylinders(xyzf_, r=0.0025)

    o3d.visualization.draw_geometries(cylinders, width=1000, height=800, window_name="cy")

    sample = np.load("C:/Users/Administrator/Desktop/crown tree/27/lines.npy")

    sample[:, 3] *= 1.5
    sample[:, 7] *= 1.5

    tree_data = lines_to_tree_structure(sample, merge_weight=0.7)

    bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
    bark_wo_uv.compute_vertex_normals()
    o3d.visualization.draw_geometries([*cylinders, bark_wo_uv], width=1000, height=800, window_name="tree")
    o3d.visualization.draw_geometries([*cylinders_, bark_wo_uv], width=1000, height=800, window_name="proj bottom")
    # o3d.visualization.draw_geometries([bark_wo_uv, leaf], width=1000, height=800, window_name="tree")

    v_xyz = sample[:, 4:7]
    v_xyz[:, 2] = xyz[0, 2].item()
    v_xyz[:, :2] *= 0.8
    v_mesh = xyz_to_mesh(v_xyz, radius=0.007, resolution=10)
    o3d.visualization.draw_geometries([*cylinders, v_mesh], width=1000, height=800, window_name="proj")

    # add leaves
    repeat_num = 2
    add_pts = np.concatenate([tree_data[:, :3]] * repeat_num, axis=0)
    fac = 0.02
    add_pts = fac * np.random.randn(add_pts.shape[0], 3) + (1 - fac) * add_pts
    new_tree_data = add_leaves_to_tree(tree_data, add_pts, th=0.01)
    bark_wo_uv_add, leaf_wo_uv_add, bark_add, leaf_add = uv_2.to_mesh(new_tree_data)
    bark_wo_uv_add.compute_vertex_normals()

    leaf_rev = deepcopy(leaf_add)
    leaf_rev_tris = np.asarray(leaf_rev.triangles)
    leaf_rev_tris = leaf_rev_tris[:, [2, 1, 0]]
    leaf_rev.triangles = o3d.utility.Vector3iVector(leaf_rev_tris)

    o3d.visualization.draw_geometries([bark, leaf_add, leaf_rev], width=1000, height=800, window_name="tree")

    np.save("lines.npy", sample)

    o3d.io.write_triangle_mesh("bark.obj", bark)
    leaf_add.compute_vertex_normals()
    o3d.io.write_triangle_mesh("leaf.obj", leaf_add)
