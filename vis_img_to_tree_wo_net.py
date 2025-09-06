import numpy as np
import torch
from copy import deepcopy
import open3d as o3d
import cv2
from utils import to_o3d_pcd, yellow
from polygon_to_tree import DDIM, renorm_w, cd, uv_2, lines_to_tree_structure, net


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
    img = cv2.imread("C:/Users/Administrator/Desktop/sketch/56.jpg")[:, :, 0]
    x = np.arange(img.shape[0]*img.shape[1]) % img.shape[1]
    z = img.shape[0] - np.arange(img.shape[0]*img.shape[1]) // img.shape[1]
    xz = np.concatenate([x[:, None], z[:, None]], axis=1)

    black_pix_inds = (img.reshape(-1) < 150)
    xz = xz[black_pix_inds]
    xyz = np.concatenate([xz[:, :1], np.zeros((xz.shape[0], 1)), xz[:, 1:]], axis=1)
    xyz = xyz[np.random.permutation(xyz.shape[0])[:4096]]
    lowest = xyz[np.argmin(xyz[:, 2]).item()]
    xyz = xyz - lowest[None, :]

    xyz = xyz / xyz[:, 2].max().item()
    xyz = xyz * 2
    xyz[:, 2] -= 1

    xz = xyz[:, [0, 2]]

    print("pix num: %d" % xyz.shape[0])
    pcd = to_o3d_pcd(xyz, [0, 0, 0])

    o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd")

    sample = np.load("C:/Users/Administrator/Desktop/sketch tree/56/lines.npy")

    sample[:, 3] *= 1.5
    sample[:, 7] *= 1.5

    try:
        tree_data = lines_to_tree_structure(sample, merge_weight=0.7, leaf_fac=15)
        bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
        bark_wo_uv.compute_vertex_normals()
        o3d.visualization.draw_geometries([pcd, bark_wo_uv], width=1000, height=800, window_name="tree")
        # o3d.visualization.draw_geometries([bark_wo_uv, leaf], width=1000, height=800, window_name="tree")

        bark_pcd_wo_y = to_o3d_pcd(np.concatenate([sample[:, 4:5], np.zeros((sample.shape[0], 1)), sample[:, 6:7]], axis=1), yellow)
        o3d.visualization.draw_geometries([pcd, bark_pcd_wo_y], width=1000, height=800, window_name="tree")

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
    except:
        print("error, next")