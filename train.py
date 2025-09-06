import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Tree, tree_collect, renorm_w
from functools import partial
from utils import processbar, render
from models.diffusion import DiffusionTree
from models.llama import LLaMAT
import math
from scipy.spatial.transform import Rotation
from visualization import line
import cv2

device = torch.device("cuda:0")


linear_start = 0.00085
linear_end = 0.0120
net = DiffusionTree(
    denoise_model=LLaMAT(depth=16, dim=768, n_heads=16, in_dim=8, out_dim=8),
    timesteps=1000, linear_start=linear_start, linear_end=linear_end,
    sample_steps=100
)
net.to(device)

epochs = 1000
base_lr = 0.0001
min_lr = 0.00001
warm_up_epoch = 0

log_imgs_dir = "./log_imgs"
save_path = "./params/llama-tree.pth"

train_dataset = Tree("G:/data10w")
batch_size = 4
log_freq = 2000
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=partial(tree_collect)
)

optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)


def update_lr(cur_epoch, epoch):
    if cur_epoch < warm_up_epoch:
        lr = base_lr * cur_epoch / warm_up_epoch
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (cur_epoch - warm_up_epoch) / (epoch - warm_up_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    print("lr update finished  cur lr: %.8f" % lr)


def log_imgs(filename):
    net.eval()
    # batch x node_num x 8
    samples = net.sample_trees_by_lms(batch=4, node_num=1024)
    samples = samples.cpu().numpy()

    imgs = []
    for i in range(samples.shape[0]):
        sample = samples[i]

        xyz1, w1, xyz2, w2 = sample[:, :3], sample[:, 3], sample[:, 4:7], sample[:, 7]
        rot_ab = Rotation.from_euler('zyx', [0, 0, -np.pi / 2]).as_matrix()
        rotated_xyz1, rotated_xyz2 = np.matmul(rot_ab, xyz1.T).T, np.matmul(rot_ab, xyz2.T).T
        w1, w2 = renorm_w(w1, train_dataset.w_min, train_dataset.w_max), renorm_w(w2, train_dataset.w_min,  train_dataset.w_max)

        lines_mesh = []
        for j in range(sample.shape[0]):
            st_pt, ed_pt = rotated_xyz1[j], rotated_xyz2[j]
            st_w, ed_w = w1[j].item(), w2[j].item()
            # print(st_pt, ed_pt, st_w, ed_w)
            lines_mesh.append(line(st_pt, ed_pt, st_w, ed_w))
        img = render(lines_mesh, w=512)
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=1)
    cv2.imwrite(filename, imgs)
    net.train()


if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        processed, iter_num = 0, 0
        loss_val = 0
        for x, masks in train_loader:
            x, masks = x.to(device), masks.to(device)

            loss = net(x, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processed += x.shape[0]
            iter_num += 1
            loss_val += loss.item()

            if iter_num % 20 == 0:
                torch.cuda.empty_cache()

            if (epoch == 1 and iter_num == 1) or iter_num % log_freq == 0:
                filename = "%s/epoch-%d-iter-%d.jpg" % (log_imgs_dir, epoch, iter_num)
                log_imgs(filename)

            print("\repoch: %d  %s  loss: %.5f  iter: %d" % (epoch, processbar(processed, len(train_dataset)), loss.item(), iter_num), end="")
        print("epoch: %d: loss: %.5f" % (epoch, loss_val))
        torch.save(net.state_dict(), save_path)
        print("save finish")
        update_lr(epoch, epochs)