import numpy as np
import torch
from copy import deepcopy
import open3d as o3d
import cv2
from utils import to_o3d_pcd, yellow
from polygon_to_tree import DDIM, renorm_w, cd, uv_2, lines_to_tree_structure, net


if __name__ == '__main__':
    img = cv2.imread("C:/Users/Administrator/Desktop/sketch/50.jpg")[:, :, 0]
    x = np.arange(img.shape[0]*img.shape[1]) % img.shape[1]
    z = img.shape[0] - np.arange(img.shape[0]*img.shape[1]) // img.shape[1]
    xz = np.concatenate([x[:, None], z[:, None]], axis=1)

    black_pix_inds = (img.reshape(-1) < 200)
    xz = xz[black_pix_inds]
    xyz = np.concatenate([xz[:, :1], np.zeros((xz.shape[0], 1)), xz[:, 1:]], axis=1)
    xyz = xyz[np.random.permutation(xyz.shape[0])[:2048]]
    lowest = xyz[np.argmin(xyz[:, 2]).item()]
    xyz = xyz - lowest[None, :]

    xyz = xyz / xyz[:, 2].max().item()
    xyz = xyz * 2
    xyz[:, 2] -= 1

    xz = xyz[:, [0, 2]]

    print("pix num: %d" % xyz.shape[0])
    pcd = to_o3d_pcd(xyz, [0, 0, 0])

    o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd")

    sampler = DDIM(net)
    # st = time.time()
    batch = 64
    while True:
        samples = sampler.loop2d(torch.Tensor(xz).cuda(), 0.001, 1024, batch, inv=True, guided_dims="xz")
        # cost = time.time() - st
        # print("%d tree: %.3fs, avg %.3f" % (batch, cost, cost / batch))
        samples = samples.cpu().numpy()
        print(samples.shape)
        for i in range(samples.shape[0]):
            sample = samples[i]
            w1, w2 = renorm_w(sample[:, 3], 0.000827, 0.046), renorm_w(sample[:, 7], 0.000827, 0.046)
            sample[:, 3] = w1
            sample[:, 7] = w2

            try:
                cd_val = cd(torch.Tensor(deepcopy(sample)[:, [4, 6]])[None, :, :].cuda(), torch.Tensor(xz)[None, :, :].cuda())
                print("cd: %.5f" % cd_val)
                if cd_val > 0.01:
                    continue

                tree_data = lines_to_tree_structure(sample, merge_weight=0.7)
                bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
                bark_wo_uv.compute_vertex_normals()
                o3d.visualization.draw_geometries([pcd, bark_wo_uv], width=1000, height=800, window_name="tree")

                leaf_rev = deepcopy(leaf)
                leaf_rev_tris = np.asarray(leaf_rev.triangles)
                leaf_rev_tris = leaf_rev_tris[:, [2, 1, 0]]
                leaf_rev.triangles = o3d.utility.Vector3iVector(leaf_rev_tris)
                o3d.visualization.draw_geometries([bark, leaf, leaf_rev], width=1000, height=800, window_name="tree")

                np.save("lines.npy", sample)
            except:
                print("error, next")