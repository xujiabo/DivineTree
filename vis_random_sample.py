import numpy as np
import open3d as o3d
import torch
from datasets import Tree, tree_collect, renorm_w
from models.diffusion import DiffusionTree
from models.llama import LLaMAT
from models.sampler import DDIM
from algorithm import uv_2
from copy import deepcopy
from utils import to_o3d_pcd, yellow, square_distance
import random
import time

device = torch.device("cuda:0")

linear_start = 0.00085
linear_end = 0.0120
net = DiffusionTree(
    denoise_model=LLaMAT(depth=16, dim=768, n_heads=16, in_dim=8, out_dim=8),
    timesteps=1000, linear_start=linear_start, linear_end=linear_end,
    sample_steps=10
)

net.to(device)
save_path = "./params/llama-tree.pth"
net.load_state_dict(torch.load(save_path))
net.eval()


def lines_to_tree_structure(lines, merge_weight=0.7, leaf_fac=15):
    # lines: n x 8
    node = np.zeros((lines.shape[0]+1, 4))
    node[0] = lines[0, :4]
    node[1] = lines[0, 4:]
    node_len = 2

    root_ed_inds = np.zeros((lines.shape[0], ), dtype=np.int64)
    root_ed_inds[0] = 1
    root_ed_inds_len = 1

    fa = np.zeros((lines.shape[0]+1, ), dtype=np.int64)
    fa[1] = 0
    other = np.ones((lines.shape[0], )).astype(np.bool8)
    other[0] = False

    while other.sum().item() > 0:
        # m x 8
        other_node = lines[other]
        root_nodes = node[root_ed_inds[:root_ed_inds_len]]
        dis = square_distance(torch.Tensor(other_node[:, :3]).unsqueeze(0), torch.Tensor(root_nodes[:, :3]).unsqueeze(0))[0]
        min_dis, min_inds = dis.min(dim=1)

        other_id = min_dis.min(dim=0)[1].item()
        other_change_id = np.nonzero(other)[0].reshape(-1)[other_id]
        other[other_change_id] = False

        node_id = root_ed_inds[min_inds[other_id].item()].item()

        selected_other_node = other_node[other_id]
        node[node_id] = node[node_id] * merge_weight + selected_other_node[:4] * (1 - merge_weight)

        node[node_len] = selected_other_node[4:]
        node_len += 1

        root_ed_inds[root_ed_inds_len] = node_len - 1
        root_ed_inds_len += 1

        fa[node_len - 1] = node_id

        print("\r%d / %d" % (lines.shape[0]-other.sum().item(), lines.shape[0]), end="")
    print()

    tree = [[] for _ in range(fa.shape[0])]
    for i in range(1, fa.shape[0]):
        fa_i = fa[i]
        tree[fa_i].append(i)

    min_w = node[:, 3].min().item()
    is_leaf = (node[:, 3] < min_w * 1.5).astype(np.float64)

    xyz = node[:, :3]
    fa[0] = -1
    fa = fa[:, None].astype(np.float64)
    width = node[:, 3:4]
    is_leaf = is_leaf[:, None]
    leaf_width = np.copy(width) * leaf_fac

    tree_data = np.concatenate([xyz, fa, width, is_leaf, leaf_width], axis=1)
    return tree_data


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # seed = 49673
    seed = 650616
    # seed = 65192511
    set_random_seed(seed)
    sampler = DDIM(net)

    st = time.time()
    batch = 4
    samples = sampler.loop(None, 0.1, 1024, batch, random_step=25)
    print("time:", (time.time() - st) / batch)

    samples = samples.cpu().numpy()
    print(samples.shape)
    for i in range(samples.shape[0]):
        sample = samples[i]
        w1, w2 = renorm_w(sample[:, 3], 0.000827, 0.046), renorm_w(sample[:, 7], 0.000827, 0.046)
        sample[:, 3] = w1
        sample[:, 7] = w2

        tree_data = lines_to_tree_structure(sample, merge_weight=0.7)
        bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(tree_data)
        bark_wo_uv.compute_vertex_normals()
        o3d.visualization.draw_geometries([bark_wo_uv], width=1000, height=800, window_name="tree")
        # o3d.visualization.draw_geometries([bark_wo_uv, leaf], width=1000, height=800, window_name="tree")

        leaf_rev = deepcopy(leaf)
        leaf_rev_tris = np.asarray(leaf_rev.triangles)
        leaf_rev_tris = leaf_rev_tris[:, [2, 1, 0]]
        leaf_rev.triangles = o3d.utility.Vector3iVector(leaf_rev_tris)

        o3d.visualization.draw_geometries([bark, leaf, leaf_rev], width=1000, height=800, window_name="tree")

        # o3d.io.write_triangle_mesh("bark.obj", bark)
        # leaf.compute_vertex_normals()
        # o3d.io.write_triangle_mesh("leaf.obj", leaf)

        # np.save("guide_pts.npy", sample[:, 4:7])