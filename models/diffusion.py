import numpy as np
import torch
from torch import nn
from models.sampler import LMS

device = torch.device("cuda:0")


class DiffusionTree(nn.Module):
    def __init__(self,
            # denoise model
            denoise_model=None,
            # noise schedule
            timesteps=1000, linear_start=0.00085, linear_end=0.0120,
            # sampler
            sample_steps=50
        ):
        super(DiffusionTree, self).__init__()
        self.timesteps = timesteps
        # noise schedule
        self.betas, self.alphas, self.alphas_cumprod, self.alphas_cumprod_prev = None, None, None, None
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = None, None
        self.register_schedule(timesteps, linear_start, linear_end)
        # denoise model (UNet)
        self.unet = denoise_model
        # loss function
        self.loss_func = nn.MSELoss(reduction="none")
        # sampler
        self.sampler = LMS(timesteps, linear_start, linear_end)
        self.sampler.set_timesteps(sample_steps)

    def register_schedule(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float32) ** 2
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas, self.alphas = betas.to(device), alphas.to(device)
        self.alphas_cumprod, self.alphas_cumprod_prev = torch.FloatTensor(alphas_cumprod).to(device), torch.FloatTensor(alphas_cumprod_prev).to(device)
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.alphas_cumprod), torch.sqrt(1. - self.alphas_cumprod)

    def add_noise_to_x0(self, x0, t, noise):
        batch_size = t.shape[0]
        xt = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1) * x0 + \
             self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1) * noise
        return xt

    def forward(self, x, mask):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()

        noise = torch.randn_like(x)
        x_noisy = self.add_noise_to_x0(x0=x, t=t, noise=noise)
        eta = self.unet(x_noisy, mask, t)

        # cal loss
        target = noise
        # batch x max_len x 8
        loss = self.loss_func(target, eta)
        loss = (loss * mask[:, :, None]).sum([1, 2])
        loss = loss / (mask.sum(1))
        return loss.mean()

    @torch.no_grad()
    def sample_trees_by_lms(self, batch=4, node_num=1024):
        # latents = torch.randn(batch, node_num, 8)  # (1, 4, 64, 64)
        latents = torch.stack([torch.randn(node_num, 8) for _ in range(batch)], dim=0)
        latents = latents * self.sampler.sigmas[0]  # sigmas[0]=157.40723
        latents = latents.to(device)
        # 循环步骤
        for i, t in enumerate(self.sampler.timesteps):  # timesteps=[999.  988.90909091 978.81818182 ...100个
            latent_model_input = latents  # (1, 4, 64, 64)
            sigma = self.sampler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
            timestamp = torch.tensor([t]*batch).cuda().float()
            noise_pred = self.unet(latent_model_input, None, timestamp)
            latents = self.sampler.step(noise_pred, i, latents)

            print("\rlms: %d / %d" % (i+1, self.sampler.num_inference_steps), end="")
        print()
        return latents.clamp(-1, 1)

    def tensors_to_lines(self, samples):
        # samples: batch x nodenum x 8
        pass


if __name__ == '__main__':
    from models.llama import LLaMAT
    device = torch.device("cuda:0")
    diff = DiffusionTree(
        denoise_model=LLaMAT(depth=12, dim=768, n_heads=16, in_dim=8, out_dim=8),
        timesteps=1000, linear_start=0.00085, linear_end=0.0120,
        sample_steps=50
    )
    diff.to(device)
    diff.eval()
    y = diff.sample_trees_by_lms(batch=8, node_num=1024)
    print(y.shape)