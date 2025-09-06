import numpy as np
import torch
from scipy import integrate
from utils import square_distance
from copy import deepcopy


class LMS:
    def __init__(self, num_train_timesteps, beta_start, beta_end):
        self.betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        return

    def set_timesteps(self, num_inference_steps=100):
        self.num_inference_steps = num_inference_steps
        # 1000：num_train_timesteps
        self.timesteps = np.linspace(1000 - 1, 0, num_inference_steps, dtype=float)
        low_idx = np.floor(self.timesteps).astype(int)
        high_idx = np.ceil(self.timesteps).astype(int)
        frac = np.mod(self.timesteps, 1.0)

        sigmas = np.array(
            ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float16)
        self.derivatives = []

    def get_lms_coefficient(self, order, t, current_order):
        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def step(self, model_output, timestep, sample):
        order = 4
        sigma = self.sigmas[timestep]
        pred_original_sample = sample - sigma * model_output
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)
        order = min(timestep + 1, order)
        lms_coeffs = [self.get_lms_coefficient(order, timestep, curr_order) for curr_order in range(order)]
        prev_sample = sample + sum(
            coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
        return prev_sample


class DDIM:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
        return a

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = torch.randn_like(x_start)

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    @torch.no_grad()
    def loop(self, condition_pcd=None, lambda1=0.5, node_num=1024, batch=8, eta=None, inv=True, random_step=50):
        device = self.diffusion.betas.device

        x_t = torch.stack([torch.randn(node_num, 8) for _ in range(batch)], dim=0).to(device)
        shape = x_t.shape
        bs = shape[0]

        betas = self.diffusion.betas
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)

        if condition_pcd is None:
            sampling_timesteps = random_step
            skip = self.diffusion.timesteps // sampling_timesteps
            seq = range(0, self.diffusion.timesteps, skip)
            seq = list(seq) + [999]
        else:
            seq = list(range(0, 400, 20)) + list(range(400, 1000, 200)) + [999]
        re_seq = list(reversed(seq))
        # seq = range(0, self.diffusion.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):

            index = re_seq.index(i) + 1
            if index < len(re_seq):
                continuous_sqrt_alpha_cumprod = torch.FloatTensor([sqrt_alphas_cumprod_prev[re_seq[index]]]).repeat(bs, 1).to(device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                if condition_pcd is not None:
                    noisy_t = self.q_sample(
                        condition_pcd.unsqueeze(0).repeat([bs, 1, 1]),
                        continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1)
                    )
            # lam = lambda1 * sqrt_alphas_cumprod[i]
            lam = lambda1

            t = (torch.ones(bs) * i).to(device)
            next_t = (torch.ones(bs) * (j)).to(device)
            et = self.diffusion.unet(x_t, None, t)
            at = self.compute_alpha(self.diffusion.betas, t.long())
            at_next = self.compute_alpha(self.diffusion.betas, next_t.long())
            x0_t = (x_t - et * (1 - at).sqrt()) / at.sqrt()
            if eta is None:
                eta = 1. if condition_pcd is not None else 0
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            if condition_pcd is None:
                x_t = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_t) + c2 * et
            else:
                mu = at_next.sqrt() * x0_t + c2 * et

                if i > 0:
                    x_t = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_t) + c2 * et
                    end_pts = x0_t[:, :, 4:7]
                    # b x m x n
                    # noisy_t = crown_pcd.unsqueeze(0).repeat([bs, 1, 1])
                    batch_condition = condition_pcd.unsqueeze(0).repeat([bs, 1, 1])
                    dis = square_distance(batch_condition, end_pts)
                    dis_inv = square_distance(end_pts, batch_condition)
                    # b x m
                    nn_inds = dis.min(dim=2)[1]
                    # b x n
                    nn_inds_inv = dis_inv.min(dim=2)[1]
                    batch_tgt = []
                    batch_tgt_inv = []

                    for k in range(batch):
                        tgt_mat = torch.scatter_add(torch.zeros(node_num, 3).to(device), 0, nn_inds[k][:, None].repeat([1, 3]), noisy_t[k])
                        cnt = torch.scatter_add(torch.zeros((node_num, )).to(device), 0, nn_inds[k], torch.ones((condition_pcd.shape[0],)).to(device))
                        tgt_mat[cnt == 0] = x_t[k, cnt == 0, 4:7]
                        cnt[cnt == 0] = 1
                        tgt_mat = tgt_mat / cnt[:, None]
                        tgt_mat = torch.cat([x_t[k, :, :4], tgt_mat, x_t[k, :, 7:]], dim=1)
                        batch_tgt.append(tgt_mat)

                        tgt_mat_inv = x_t[k].clone()
                        tgt_mat_inv[:, 4:7] = noisy_t[k, nn_inds_inv[k]]
                        batch_tgt_inv.append(tgt_mat_inv)

                    batch_tgt = torch.stack(batch_tgt, dim=0)
                    batch_tgt_inv = torch.stack(batch_tgt_inv, dim=0)

                    if inv:
                        batch_tgt = (batch_tgt + batch_tgt_inv) / 2
                    noisy_t = batch_tgt
                    x_t = (lam * x_t + noisy_t) / (1 + lam)
                else:
                    x_t = mu
            print("\rddim: %d / %d" % (i, self.diffusion.timesteps), end="")
        print()
        return x_t

    @torch.no_grad()
    def loop2d(self, points2d=None, lambda1=0.5, node_num=1024, batch=8, eta=None, inv=True, guided_dims="xy"):
        # crown: m x 3
        device = self.diffusion.betas.device

        x_t = torch.stack([torch.randn(node_num, 8) for _ in range(batch)], dim=0).to(device)
        shape = x_t.shape
        bs = shape[0]

        betas = self.diffusion.betas
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        dims = np.array([{"x": 4, "y": 5, "z": 6}[x] for x in guided_dims]).astype(np.int64)
        not_modify_dims = np.setdiff1d(np.arange(8), dims).astype(np.int64)
        dims, not_modify_dims = dims.tolist(), not_modify_dims.tolist()

        if points2d is None:
            sampling_timesteps = 10
            skip = self.diffusion.timesteps // sampling_timesteps
            seq = range(0, self.diffusion.timesteps, skip)
            seq = list(seq) + [999]
        else:
            seq = list(range(0, 400, 20)) + list(range(400, 1000, 200)) + [999]
        re_seq = list(reversed(seq))
        # seq = range(0, self.diffusion.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):

            index = re_seq.index(i) + 1
            if index < len(re_seq):
                continuous_sqrt_alpha_cumprod = torch.FloatTensor([sqrt_alphas_cumprod_prev[re_seq[index]]]).repeat(bs, 1).to(device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                if points2d is not None:
                    noisy_t = self.q_sample(
                        points2d.unsqueeze(0).repeat([bs, 1, 1]),
                        continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1)
                    )
            lam = lambda1
            t = (torch.ones(bs) * i).to(device)
            next_t = (torch.ones(bs) * (j)).to(device)
            et = self.diffusion.unet(x_t, None, t)
            at = self.compute_alpha(self.diffusion.betas, t.long())
            at_next = self.compute_alpha(self.diffusion.betas, next_t.long())
            x0_t = (x_t - et * (1 - at).sqrt()) / at.sqrt()
            if eta is None:
                eta = 1. if points2d is not None else 0
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            if points2d is None:
                x_t = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_t) + c2 * et
            else:
                mu = at_next.sqrt() * x0_t + c2 * et
                var = c1

                if i > 0:
                    x_t = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_t) + c2 * et
                    end_pts = x0_t[:, :, dims]
                    # b x m x n
                    batch_condition = points2d.unsqueeze(0).repeat([bs, 1, 1])
                    dis = square_distance(batch_condition, end_pts)
                    dis_inv = square_distance(end_pts, batch_condition)
                    # b x m
                    nn_inds = dis.min(dim=2)[1]
                    # b x n
                    nn_inds_inv = dis_inv.min(dim=2)[1]
                    batch_tgt = []
                    batch_tgt_inv = []

                    for k in range(batch):
                        tgt_mat = torch.scatter_add(torch.zeros(node_num, len(dims)).to(device), 0, nn_inds[k][:, None].repeat([1, len(dims)]), noisy_t[k])
                        cnt = torch.scatter_add(torch.zeros((node_num,)).to(device), 0, nn_inds[k], torch.ones((points2d.shape[0],)).to(device))
                        tgt_mat[cnt == 0] = x_t[k, cnt == 0][:, dims]
                        cnt[cnt == 0] = 1
                        tgt_mat = tgt_mat / cnt[:, None]
                        tgt_mat_ = x_t[k].clone()
                        tgt_mat_[:, dims] = tgt_mat
                        # tgt_mat = torch.cat([img[k, :, :4], tgt_mat, img[k, :, 7:]], dim=1)
                        batch_tgt.append(tgt_mat_)

                        tgt_mat_inv = x_t[k].clone()
                        tgt_mat_inv[:, dims] = noisy_t[k, nn_inds_inv[k]]
                        batch_tgt_inv.append(tgt_mat_inv)

                    batch_tgt = torch.stack(batch_tgt, dim=0)
                    batch_tgt_inv = torch.stack(batch_tgt_inv, dim=0)

                    if inv:
                        batch_tgt = (batch_tgt + batch_tgt_inv) / 2

                    noisy_t = batch_tgt
                    x_t = (lam * x_t + noisy_t) / (1 + lam)
                else:
                    x_t = mu
            print("\rddim: %d / %d" % (i, self.diffusion.timesteps), end="")
        print()
        return x_t
