import numpy as np
import open3d as o3d
import torch
from datasets import Tree, tree_collect, renorm_w
from models.diffusion import DiffusionTree
from models.llama import LLaMAT
from polygon_to_tree import DDIM, renorm_w, cd, uv_2, lines_to_tree_structure, net
from algorithm import uv_2
from copy import deepcopy
from utils import to_o3d_pcd, yellow

device = torch.device("cuda:0")


if __name__ == '__main__':
    sampler = DDIM(net)

    pcd_xyz = np.asarray(o3d.io.read_point_cloud("G:/adtree_data/70.xyz").points)
    pcd_xyz = pcd_xyz[np.random.permutation(pcd_xyz.shape[0])[:8192]]
    lowest = pcd_xyz[np.argmin(pcd_xyz[:, 2]).item()]
    pcd_xyz = pcd_xyz[pcd_xyz[:, 2] > lowest[2] + 0]
    lowest = pcd_xyz[np.argmin(pcd_xyz[:, 2]).item()]
    pcd_xyz = pcd_xyz - lowest[None, :]
    pcd_xyz = pcd_xyz / pcd_xyz[:, 2].max().item()
    pcd_xyz = pcd_xyz * 2
    pcd_xyz[:, 2] -= 1
    pcd = to_o3d_pcd(pcd_xyz, yellow)
    o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd")

    # samples = sampler.loop(None, 0.1, 1024, 8)
    batch = 64
    while True:
        samples = sampler.loop(torch.Tensor(pcd_xyz).cuda(), 0.001, 1024, batch, inv=True)
        samples = samples.cpu().numpy()
        print(samples.shape)
        for i in range(samples.shape[0]):
            sample = samples[i]
            w1, w2 = renorm_w(sample[:, 3], 0.000827, 0.046), renorm_w(sample[:, 7], 0.000827, 0.046)
            sample[:, 3] = w1
            sample[:, 7] = w2

            try:
                cd_val = cd(torch.Tensor(deepcopy(sample)[:, [4, 5, 6]])[None, :, :].cuda(), torch.Tensor(pcd_xyz)[None, :, :].cuda())
                print("cd: %.5f" % cd_val)
                if cd_val > 0.015:
                    continue

                tree_data = lines_to_tree_structure(sample, merge_weight=0.7)
                bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
                bark_wo_uv.compute_vertex_normals()
                o3d.visualization.draw_geometries([pcd, bark_wo_uv], width=1000, height=800, window_name="tree")
                # o3d.visualization.draw_geometries([bark_wo_uv, leaf], width=1000, height=800, window_name="tree")

                leaf_rev = deepcopy(leaf)
                leaf_rev_tris = np.asarray(leaf_rev.triangles)
                leaf_rev_tris = leaf_rev_tris[:, [2, 1, 0]]
                leaf_rev.triangles = o3d.utility.Vector3iVector(leaf_rev_tris)

                o3d.visualization.draw_geometries([bark, leaf, leaf_rev], width=1000, height=800, window_name="tree")

                # o3d.io.write_triangle_mesh("bark.obj", bark)
                # leaf.compute_vertex_normals()
                # o3d.io.write_triangle_mesh("leaf.obj", leaf)
                np.save("lines.npy", sample)
            except:
                print("error, next")
        # np.save("guide_pts.npy", sample[:, 4:7])