import numpy as np
import open3d as o3d
import torch
import cv2

blue, yellow, gray = [0, 0.651, 0.929], [1, 0.706, 0], [0.752, 0.752, 0.752]


def processbar(current, totle):
    process_str = ""
    for i in range(int(20 * current / totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|   %d / %d" % (process_str, current, totle)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
    return pcd


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # farthest = torch.zeros((B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        print("\rfps process: %d / %d" % (i+1, npoint), end="")
    print()
    return centroids


def render(mesh_list, w=512):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=w)

    for mesh in mesh_list:
        vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("render_dem.jpg")
    img = cv2.imread("render_dem.jpg")
    vis.destroy_window()
    return img


def lines_to_tree_structure(lines, merge_weight=0.7):
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
    leaf_width = np.copy(width) * 15

    tree_data = np.concatenate([xyz, fa, width, is_leaf, leaf_width], axis=1)
    return tree_data


def get_cylinders(data, r=0.0025):
    # n x 4 (xyz fa)
    pts = data[:, :3]
    lines = []
    for i in range(1, data.shape[0]):
        st, ed = pts[i], pts[int(data[i, 3].item())]
        length = np.linalg.norm(st-ed, axis=0).item()
        if length == 0:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(r, length, resolution=4)
        cyl.paint_uniform_color(blue)
        P = np.array([
            [0, 0, np.linalg.norm(st-ed).item()/2],
            [0, 0, 0],
            [0, 0, -np.linalg.norm(st-ed).item()/2],
        ])
        Q = np.array([
            st.tolist(),
            ((st+ed)/2).tolist(),
            ed.tolist()
        ])
        p_, q_ = P.mean(axis=0).reshape(1, -1), Q.mean(axis=0).reshape(1, -1)
        p, q = P-p_, Q-q_
        H = p.T.dot(q)
        U, Sigma, V = np.linalg.svd(H, compute_uv=True)
        V = V.T

        R = V.dot(U.T)

        v_neg = np.copy(V)
        v_neg[:, 2] = v_neg[:, 2] * -1
        rot_mat_neg = v_neg @ U.T

        R = R if np.linalg.det(R) > 0 else rot_mat_neg

        cyl.rotate(R)
        cyl.translate((st+ed)/2)

        lines.append(cyl)

    return lines