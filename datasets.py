import numpy as np
import torch
from torch.utils import data
from scipy.spatial.transform import Rotation


def norm_tree_by_height(tree):
    xyz = tree[:, :3]
    w, lw = tree[:, 4], tree[:, 6]
    h = xyz[:, 2].max().item()
    tree[:, :3] = xyz * 2 / h
    tree[:, 2] -= 1
    tree[:, 4] = w * 2 / h
    tree[:, 6] = lw * 2 / h


def less_than(node1, node2):
    if node1[2] != node2[2]:
        return (node2[2] - node2[2]) < 0
    if node1[1] != node2[1]:
        return (node2[1] - node2[1]) < 0
    return (node2[0] - node2[0]) < 0


def dfs(tree_arr, fa_id, cur_id, lines, xyz):
    if fa_id != -1:
        fa_xyz, cur_xyz = xyz[fa_id], xyz[cur_id]
        length = np.linalg.norm(cur_xyz-fa_xyz).item()
        if length != 0:
            lines.append([fa_id, cur_id])

    children_ids = tree_arr[cur_id]
    if len(children_ids) > 2:
        print("error !  binary tree's children number can't > 2 !  check !!!")
    if len(children_ids) == 1:
        dfs(tree_arr, cur_id, children_ids[0], lines, xyz)
    if len(children_ids) == 2:
        ch1, ch2 = xyz[children_ids[0]], xyz[children_ids[1]]
        if less_than(ch1, ch2):
            left_id, right_id = children_ids[0], children_ids[1]
        else:
            left_id, right_id = children_ids[1], children_ids[0]
        dfs(tree_arr, cur_id, left_id, lines, xyz)
        dfs(tree_arr, cur_id, right_id, lines, xyz)


def tree_to_lines(tree):
    xyz, fa, width, is_leaf, leaf_size = tree[:, :3], tree[:, 3], tree[:, 4], tree[:, 5], tree[:, 6]
    fa = fa.astype(np.int64)
    tree = [[] for _ in range(xyz.shape[0])]
    for i in range(1, xyz.shape[0]):
        fa_id = fa[i]
        tree[fa_id].append(i)
    lines = []
    dfs(tree, -1, 0, lines, xyz)
    lines = np.array(lines).astype(np.int64)
    return lines


def norm_w(normed_tree, w_min, w_max):
    width = normed_tree[:, 4]
    width = (width - w_min) / (w_max - w_min)
    width = width * 2 - 1
    normed_tree[:, 4] = width


def renorm_w(width, w_min, w_max):
    width = (width + 1) / 2
    width = width * (w_max - w_min) + w_min
    return width


def rotate_tree(tree):
    xyz = tree[:, :3]
    euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
    euler_ab[1] = 0
    euler_ab[2] = 0
    rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
    # print(rot_ab)
    rotated_xyz = np.matmul(rot_ab, xyz.T).T
    tree[:, :3] = rotated_xyz


def tree_to_line_mesh(data):
    lines = tree_to_lines(data)
    lines_mesh = []
    for j in range(lines.shape[0]):
        st_ind, ed_ind = lines[j].tolist()
        st_pt, ed_pt = xyz[st_ind], xyz[ed_ind]
        st_w, ed_w = width[st_ind], width[ed_ind]
        # print(st_pt, ed_pt, st_w, ed_w)
        lines_mesh.append(line(st_pt, ed_pt, st_w, ed_w))
    return lines_mesh


def vis_tree_as_lines(data):
    lines_mesh = tree_to_line_mesh(data)
    o3d.draw_geometries(lines_mesh, height=800, width=1000, window_name="lines tree")


class Tree(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.w_min = 0.000827
        self.w_max = 0.046 #* 2

    def __len__(self):
        return 100000

    def __getitem__(self, item):
        path = "%s/tree%d.npy" % (self.root, item)
        tree_data = np.load(path)
        norm_tree_by_height(tree_data)
        # [1, 2.1)
        w_fac = (np.random.rand(1)*1.1 + 1).item()
        # print(w_fac)
        tree_data[:, 4] *= w_fac
        # vis_tree_as_lines(tree_data)

        norm_w(tree_data, self.w_min, self.w_max)
        # xyz, w = tree_data[:, :3], tree_data[:, 4]
        # print(xyz.min(axis=0))
        # print(xyz.max(axis=0))
        # print(w.min(), w.max())

        # random rotate
        rotate_tree(tree_data)

        return tree_data


def tree_collect(batch):
    max_line_num = -1
    tree_lines = []
    for ind, tree_data in enumerate(batch):
        lines_ind = tree_to_lines(tree_data)
        max_line_num = max(max_line_num, lines_ind.shape[0])
        xyz, w = tree_data[:, :3], tree_data[:, 4:5]
        lines_st = np.concatenate([xyz[lines_ind[:, 0]], w[lines_ind[:, 0]]], axis=1)
        lines_ed = np.concatenate([xyz[lines_ind[:, 1]], w[lines_ind[:, 1]]], axis=1)
        lines = np.concatenate([lines_st, lines_ed], axis=1)
        tree_lines.append(lines)
    masks = []
    for i in range(len(tree_lines)):
        mask = [1] * tree_lines[i].shape[0]
        if tree_lines[i].shape[0] < max_line_num:
            mask = mask + [0] * (max_line_num - tree_lines[i].shape[0])
            tree_lines[i] = np.concatenate([tree_lines[i], np.zeros((max_line_num - tree_lines[i].shape[0], 8))], axis=0)
        masks.append(mask)
    # batch x max_len
    masks = torch.Tensor(masks)
    # batch x max_len x 8 (xyzwxyzw)
    batched_tree_lines = np.stack(tree_lines, axis=0)
    batched_tree_lines = torch.from_numpy(batched_tree_lines).float()
    return batched_tree_lines, masks


if __name__ == '__main__':
    from visualization import line
    import open3d as o3d
    from utils import render
    ds = Tree("G:/data10w")
    # w_min, w_max = 1000, -1
    # lw_min, lw_max = 1000, -1
    # for i in range(len(ds)):
    #     data = ds[i]
    #     xyz, fa, width, is_leaf, leaf_size = data[:, :3], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
    #     w_min = min(w_min, width.min().item())
    #     w_max = max(w_max, width.max().item())
    #     print("\r%d / %d" % (i+1, len(ds)), end="")
    # print()
    # print("w_min: %.8f  w_max: %.8f" % (w_min, w_max))

    for i in range(len(ds)):
        data = ds[i]
        xyz, fa, width, is_leaf, leaf_size = data[:, :3], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
        w = renorm_w(width, ds.w_min, ds.w_max)
        data[:, 4] = w

        rot_ab = Rotation.from_euler('zyx', [0, 0, -np.pi/2]).as_matrix()
        # print(rot_ab)
        rotated_xyz = np.matmul(rot_ab, xyz.T).T
        data[:, :3] = rotated_xyz
        lines_mesh = tree_to_line_mesh(data)
        render(lines_mesh, w=512)