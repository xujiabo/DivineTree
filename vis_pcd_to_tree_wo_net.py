import numpy as np
import open3d as o3d
import torch
from datasets import Tree, tree_collect, renorm_w
from models.diffusion import DiffusionTree
from models.llama import LLaMAT
from models.sampler import DDIM
from vis_random_sample import lines_to_tree_structure
from algorithm import uv_2
from copy import deepcopy
from utils import to_o3d_pcd, yellow


def xyz_to_mesh(xyz, r=0.008):
    pts, edges = [], []
    for i in range(xyz.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=10)

        np.asarray(sphere.vertices)[:] += xyz[i]

        pts.append(np.asarray(sphere.vertices))
        edges.append(np.asarray(sphere.triangles)+np.asarray(sphere.vertices).shape[0]*i)
    pts = np.concatenate(pts, axis=0)
    edges = np.concatenate(edges, axis=0)
    print(pts.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(edges)
    mesh.compute_vertex_normals()
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
    # pcd_xyz = np.asarray(o3d.io.read_point_cloud("G:/adtree_data/2.xyz").points)
    pcd_xyz = np.asarray(o3d.io.read_point_cloud("a_demo/2.xyz").points)

    lowest = pcd_xyz[np.argmin(pcd_xyz[:, 2]).item()]
    pcd_xyz = pcd_xyz[pcd_xyz[:, 2] > lowest[2] + 0]
    lowest = pcd_xyz[np.argmin(pcd_xyz[:, 2]).item()]
    pcd_xyz = pcd_xyz - lowest[None, :]
    pcd_xyz = pcd_xyz / pcd_xyz[:, 2].max().item()
    pcd_xyz = pcd_xyz * 2
    pcd_xyz[:, 2] -= 1

    pcd_xyz = pcd_xyz[np.random.permutation(pcd_xyz.shape[0])[:8192]]
    print(pcd_xyz.shape)

    mesh = xyz_to_mesh(pcd_xyz, r=0.01)
    crown_pcd = to_o3d_pcd(pcd_xyz, yellow)

    # sample_path = "C:/Users/Administrator/Desktop/als tree/2/lines.npy"
    sample_path = "a_demo/lines.npy"
    sample = np.load(sample_path)

    sample[:, 3] *= 1
    sample[:, 7] *= 1

    tree_data = lines_to_tree_structure(sample, merge_weight=0.7)
    # tree_data[:, 6] *= 1.5
    bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
    bark_wo_uv.compute_vertex_normals()
    # o3d.visualization.draw_geometries([crown_pcd, bark_wo_uv], width=1000, height=800, window_name="tree")
    o3d.visualization.draw_geometries([mesh, bark_wo_uv], width=1000, height=800, window_name="tree")
    o3d.visualization.draw_geometries([mesh], width=1000, height=800, window_name="tree")
    o3d.visualization.draw_geometries([bark_wo_uv], width=1000, height=800, window_name="tree")

    # add leaves
    repeat_num = 2
    add_pts = np.concatenate([tree_data[:, :3]]*repeat_num, axis=0)
    fac = 0.02
    add_pts = fac * np.random.randn(add_pts.shape[0], 3) + (1-fac)*add_pts
    new_tree_data = add_leaves_to_tree(tree_data, add_pts, th=0.01)
    bark_wo_uv_add, leaf_wo_uv_add, bark_add, leaf_add = uv_2.to_mesh(new_tree_data)
    bark_wo_uv_add.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh, bark_wo_uv_add], width=1000, height=800, window_name="tree add")

    # add leaf
    leaf_rev = deepcopy(leaf_add)
    leaf_rev_tris = np.asarray(leaf_rev.triangles)
    leaf_rev_tris = leaf_rev_tris[:, [2, 1, 0]]
    leaf_rev.triangles = o3d.utility.Vector3iVector(leaf_rev_tris)

    o3d.visualization.draw_geometries([bark, leaf_add, leaf_rev], width=1000, height=800, window_name="tree")

    # o3d.io.write_triangle_mesh("bark.obj", bark)
    # leaf_add.compute_vertex_normals()
    # o3d.io.write_triangle_mesh("leaf.obj", leaf_add)

    # o3d.io.write_triangle_mesh("pcd.obj", mesh)