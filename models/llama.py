import torch
from torch import nn
import math
from torch.nn import functional as F


# RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_weight=True):
        super().__init__()
        self.eps = eps
        self.use_weight = use_weight
        if use_weight:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.use_weight:
            output = output * self.weight
        return output


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore

    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''torch.repeat_interleave(x, dim=2, repeats=n_rep)'''
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        # 将输入张量在第四个维度上扩展 n_rep 次
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # 调整为适当的形状
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    '''Multi-head attention module.'''

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_kv_heads = n_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads
        # Q的头数* head_dim
        # self.wq = ColumnParallelLinear(args.dim,args.n_heads * self.head_dim,bias=False,gather_output=False,init_method=lambda x: x,)
        self.wq = nn.Linear(dim, n_heads*self.head_dim, bias=False)
        # K的头数* head_dim
        # self.wk = ColumnParallelLinear(args.dim,self.n_kv_heads * self.head_dim,bias=False, gather_output=False,init_method=lambda x: x,)
        self.wk = nn.Linear(dim, self.n_kv_heads*self.head_dim, bias=False)
        # V的头数* head_dim
        # self.wv = ColumnParallelLinear(args.dim,self.n_kv_heads * self.head_dim,bias=False,gather_output=False,init_method=lambda x: x,)
        self.wv = nn.Linear(dim, self.n_kv_heads*self.head_dim, bias=False)
        # self.wo = RowParallelLinear(args.n_heads * self.head_dim,args.dim,bias=False,input_is_parallel=True,init_method=lambda x: x,)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 对当前输入的query和key进行RoPE，注意kv_cache里面的key已经做过了RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim
        )
        self.attention_norm = RMSNorm(dim, eps=1e-5)
        self.ffn_norm = RMSNorm(dim, eps=1e-5)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: torch.Tensor,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLaMA(nn.Module):
    def __init__(self, depth, dim, n_heads, in_dim, out_dim, max_len=1200):
        super(LLaMA, self).__init__()
        self.layers = torch.nn.ModuleList()
        for layer_id in range(depth):
            self.layers.append(TransformerBlock(dim, n_heads))

        self.inp = nn.Linear(in_dim, dim)
        self.norm = RMSNorm(dim, eps=1e-5)
        self.output = nn.Linear(dim, out_dim)

        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_len * 2).to(torch.device("cuda:0"))

    def forward(self, x, mask):
        # batch x l x d_in, batch x seqlen
        b, seqlen = x.shape[0], x.shape[1]
        freqs_cis = self.freqs_cis[:seqlen]
        if mask is not None:
            # b x seqlen x seqlen
            mask = (mask[:, None, :].repeat([1, seqlen, 1]) + mask[:, :, None].repeat([1, 1, seqlen]) < 2).float() * -1e9
            mask = mask[:, None, :, :]

        x = self.inp(x)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        x = self.norm(x)
        output = self.output(x)
        return output

########## Diffusion UNet ##########

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TransformerBlockT(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim
        )
        self.attention_norm = RMSNorm(dim, eps=1e-6, use_weight=False)
        self.ffn_norm = RMSNorm(dim, eps=1e-6, use_weight=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: torch.Tensor,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis, mask)
        x = x + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
        return x


class LLaMAT(nn.Module):
    def __init__(self, depth, dim, n_heads, in_dim, out_dim, max_len=1200):
        super(LLaMAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        for layer_id in range(depth):
            self.layers.append(TransformerBlockT(dim, n_heads))

        self.t_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

        self.inp = nn.Linear(in_dim, dim)
        self.norm_final = RMSNorm(dim, eps=1e-6, use_weight=False)
        self.output = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_len * 2).to(torch.device("cuda:0"))
        self.initialize_weights()

    def forward(self, x, mask, t):
        # batch x l x d_in, batch x seqlen
        b, seqlen, d_in = x.shape[0], x.shape[1], x.shape[2]
        freqs_cis = self.freqs_cis[:seqlen]
        if mask is not None:
            # b x seqlen x seqlen
            mask = (mask[:, None, :].repeat([1, seqlen, 1]) + mask[:, :, None].repeat([1, 1, seqlen]) < 2).float() * -1e9
            mask = mask[:, None, :, :]
        t = self.t_mlp(t)
        # tokenizer
        x = self.inp(x)
        # attention
        for i, layer in enumerate(self.layers):
            x = layer(x, t, freqs_cis, mask)

        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.output(x)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.inp.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.inp.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_mlp[1].weight, std=0.02)
        nn.init.normal_(self.t_mlp[3].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    # mask = torch.Tensor(
    #     [[1, 1, 1, 0, 0],
    #     [1, 1, 1, 1, 0]]
    # )
    # seqlen = mask.shape[1]
    # mask = mask[:, None, :].repeat([1, seqlen, 1]) + mask[:, :, None].repeat([1, 1, seqlen])
    # mask = (mask < 2).float() * -1e9
    # print(mask)

    # net = LLaMA(depth=12, dim=512, n_heads=16, in_dim=8, out_dim=8)
    # net.to(device)
    # inp = torch.randn(2, 1024, 8).to(device)
    # out = net(inp, None)
    # print(out.shape)


    mask = torch.Tensor(
        [[1]*1000+[0]*24,
         [1]*1023+[0]]
    ).to(device)
    print(mask.shape)
    t = torch.Tensor([256, 452]).to(device)

    net = LLaMAT(depth=12, dim=512, n_heads=16, in_dim=8, out_dim=8)
    net.to(device)
    inp = torch.randn(2, 1024, 8).to(device)

    y = net(inp, mask, t)
    print(y.shape)