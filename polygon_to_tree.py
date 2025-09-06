import numpy as np
import torch
import open3d as o3d
from utils import get_cylinders, to_o3d_pcd, yellow, square_distance
from models.sampler import DDIM
from vis_random_sample import lines_to_tree_structure, net
from datasets import renorm_w
from algorithm import uv_2
from copy import deepcopy
from torch import nn


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


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, f, f_):
        if f.shape[1] == 0:
            return 0
        try:
            dis = square_distance(f, f_)
            dis[dis <= 0] = 0
            f2f_, f_2f = dis.min(dim=2)[0], dis.min(dim=1)[0]
            d = f2f_.mean(dim=1) + f_2f.mean(dim=1)
        except:
            print(f.shape, f_.shape)
        return d.mean()


cd = ChamferLoss()


if __name__ == '__main__':
    import time
    # 9  14  15  19  23  24  27  28
    poly = np.load("C:/Users/Administrator/Desktop/match_img_pcd/polygons/35.npy")
    xyz = np.concatenate([poly, -np.ones((poly.shape[0], 1))], axis=1)
    xyz_center = np.mean(xyz, axis=0, keepdims=True)
    xyz[:, :2] = xyz[:, :2] - xyz_center[:, :2]
    radius = np.linalg.norm(xyz[:, :2], axis=1).max().item()
    xyz[:, :2] = xyz[:, :2] / radius * 0.4
    xyz[:, 2] += 1.8

    fa = [poly.shape[0]-1]+np.arange(poly.shape[0]-1).tolist()
    xyzf = np.concatenate([xyz, np.array(fa)[:, None]], axis=1)
    cylinders = get_cylinders(xyzf, r=0.0025)

    o3d.visualization.draw_geometries(cylinders, width=1000, height=800, window_name="cy")

    sampler = DDIM(net)
    st = time.time()
    batch = 96
    while True:
        samples = sampler.loop2d(torch.Tensor(xyz[:, :2]).cuda(), 0.001, 1024, batch, inv=True, guided_dims="xy")
        cost = time.time() - st
        print("%d tree: %.3fs, avg %.3f" % (batch, cost, cost/batch))
        samples = samples.cpu().numpy()
        print(samples.shape)
        for i in range(samples.shape[0]):
            sample = samples[i]
            w1, w2 = renorm_w(sample[:, 3], 0.000827, 0.046), renorm_w(sample[:, 7], 0.000827, 0.046)
            sample[:, 3] = w1
            sample[:, 7] = w2

            try:
                cd_val = cd(torch.Tensor(deepcopy(sample)[:, 4:6])[None, :, :].cuda(), torch.Tensor(xyz[:, :2])[None, :, :].cuda())
                print("cd: %.5f" % cd_val)
                if cd_val > 0.0035:
                    continue
                tree_data = lines_to_tree_structure(sample, merge_weight=0.7)
                bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
                bark_wo_uv.compute_vertex_normals()
                o3d.visualization.draw_geometries([*cylinders, bark_wo_uv], width=1000, height=800, window_name="tree")
                # o3d.visualization.draw_geometries([bark_wo_uv, leaf], width=1000, height=800, window_name="tree")

                v_xyz = deepcopy(sample)[:, 4:7]
                v_xyz[:, 2] = xyz[0, 2].item()
                v_mesh = xyz_to_mesh(v_xyz, radius=0.007, resolution=10)
                o3d.visualization.draw_geometries([*cylinders, v_mesh], width=1000, height=800, window_name="proj")

                leaf_rev = deepcopy(leaf)
                leaf_rev_tris = np.asarray(leaf_rev.triangles)
                leaf_rev_tris = leaf_rev_tris[:, [2, 1, 0]]
                leaf_rev.triangles = o3d.utility.Vector3iVector(leaf_rev_tris)

                o3d.visualization.draw_geometries([bark, leaf, leaf_rev], width=1000, height=800, window_name="tree")
                np.save("lines.npy", sample)
            except:
                print("error, next")
