"""
FlowDCN model wrapper for ImageNet 256x256.
Uses VAE for latent decoding and supports classifier-free guidance.
"""

import math
import torch
import torch.nn as nn
from torch.nn.init import zeros_

from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from noise_schedulers import get_scheduler, make_interpolant_converter
from samplers.general_solver import ODESolver

ODE_MAPPER = {
    "euler": "v_prediction", "midpoint": "v_prediction",
    "ipndm": "noise_prediction", "uni_pc": "data_prediction",
    "rk45": "v_prediction",
}


def _model_wrapper(model, guidance_scale=1.0, use_checkpoint=False):
    """Wraps FlowDCN model, always returns velocity prediction."""
    def v_pred_fn(x, t_continuous, condition=None):
        if use_checkpoint:
            output = checkpoint(model, x, t_continuous, condition)
        else:
            output = model(x, t_continuous, condition)
        return output

    def model_fn(x, _, t_continuous, condition=None, unconditional_condition=None, *args, **kwargs):
        gs = guidance_scale
        if gs > 1.0:
            x = torch.cat([x, x], dim=0)
            c = torch.cat([condition, unconditional_condition], dim=0)
            t = torch.cat([t_continuous, t_continuous], dim=0)
            v_preds = v_pred_fn(x, t, c)
            # CFG on first 3 channels only (following SiT convention)
            eps, rest = v_preds[:, :3], v_preds[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            cond_rest, uncond_rest = torch.split(rest, len(rest) // 2, dim=0)
            guided_eps = uncond_eps + gs * (cond_eps - uncond_eps)
            v_pred = torch.cat([guided_eps, cond_rest], dim=1)
        else:
            v_pred = v_pred_fn(x, t_continuous, condition)
        return v_pred.to(torch.float32)

    return model_fn


def get_pretrained_flowdcn_model(args, requires_grad=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = OmegaConf.load(args.config)

    latent_size = 32
    net = FlowDCN(
        input_size=config.input_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_groups=config.num_groups,
        hidden_size=config.hidden_size,
        num_blocks=config.num_blocks,
        num_classes=config.num_classes,
        learn_sigma=config.learn_sigma,
    )
    weight = torch.load(args.ckp_path, map_location=torch.device('cpu'))
    if "optimizer_states" in weight.keys():
        weight = weight["optimizer_states"][0]["ema"]
        params = list(net.parameters())
        for w, p in zip(weight, params):
            p.data.copy_(w)
    else:
        net.load_state_dict(weight)
    net.eval()
    net = net.to(device)

    if not requires_grad:
        for param in net.parameters():
            param.requires_grad = False

    original_schedule = get_scheduler(args.schedule)()
    new_schedule = None
    if getattr(args, 'new_schedule', None) is not None and "si" in args.new_schedule:
        new_schedule = get_scheduler(args.new_schedule)(
            p_order=args.p_order, orig_sched=original_schedule
        )

    model_fn = _model_wrapper(net, guidance_scale=args.scale,
                               use_checkpoint=getattr(args, 'low_gpu', False))

    if new_schedule is not None:
        solver = ODESolver(original_schedule, ODE_MAPPER.get(args.solver_name, "v_prediction"),
                           args.prediction_type)
        model_fn_final = make_interpolant_converter(
            original_schedule, new_schedule, model_fn, solver
        )
    else:
        model_fn_final = model_fn
        new_schedule = original_schedule

    # VAE for latent decoding
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    def _decoding_fn(x):
        returning_traj = False
        if isinstance(x, tuple):
            returning_traj = True
            x, aux_data = x
        latents = x.to(dtype=vae.dtype)
        image = vae.decode(latents / 0.18215).sample
        if image.min() >= 0 and image.max() <= 1:
            image = image * 2 - 1
        if returning_traj:
            return image, aux_data
        return image

    return (model_fn_final, net, _decoding_fn,
            original_schedule, new_schedule,
            latent_size, 4, config.height, 3)


# ──────────────────────────────────────────────────────────────
# FlowDCN architecture (from FlowDCN repo)
# ──────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale[:, None, None]) + shift[:, None, None]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(next(self.mlp.parameters()).dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.norm_final(x)
        return self.linear(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.view(b, h * w, c)
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x.view(b, h, w, c)


class MultiScaleDCN(nn.Module):
    def __init__(self, in_channels, groups, channels, kernels, deformable_biass=True):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.channels = channels
        self.kernels = kernels
        self.v = nn.Linear(in_channels, groups * channels, bias=True)
        self.qk_deformables = nn.Linear(in_channels, groups * kernels * 2, bias=True)
        self.qk_scales = nn.Linear(in_channels, groups * kernels, bias=False)
        self.qk_weights = nn.Linear(in_channels, groups * kernels, bias=True)
        self.out = nn.Linear(groups * channels, in_channels)
        self.deformables_prior = nn.Parameter(torch.randn((1, 1, 1, 1, kernels, 2)), requires_grad=False)
        self.deformables_scale = nn.Parameter(torch.ones((1, 1, 1, groups, 1, 1)), requires_grad=True)
        self.max_scale = 6
        self._init_weights()

    def _init_weights(self):
        zeros_(self.qk_deformables.weight.data)
        zeros_(self.qk_scales.weight.data)
        zeros_(self.qk_deformables.bias.data)
        zeros_(self.qk_weights.weight.data)
        zeros_(self.v.bias.data)
        zeros_(self.out.bias.data)
        num_prior = int(self.kernels ** 0.5)
        dx = torch.linspace(-1, 1, num_prior, device="cuda")
        dy = torch.linspace(-1, 1, num_prior, device="cuda")
        dxy = torch.meshgrid([dx, dy], indexing="xy")
        dxy = torch.stack(dxy, dim=-1).view(-1, 2)
        self.deformables_prior.data[..., :num_prior * num_prior, :] = dxy
        for i in range(self.groups):
            scale = (i + 1) / self.groups - 0.0001
            inv_scale = math.log((scale) / (1 - scale))
            self.deformables_scale.data[..., i, :, :] = inv_scale

    def forward(self, x):
        from models.flowdcn_ops.triton_kernels.function import DCNFunction
        B, H, W, _ = x.shape
        v = self.v(x).view(B, H, W, self.groups, self.channels)
        deformables = self.qk_deformables(x).view(B, H, W, self.groups, self.kernels, 2)
        scale = self.qk_scales(x).view(B, H, W, self.groups, self.kernels, 1) + self.deformables_scale
        deformables = (deformables + self.deformables_prior) * scale.sigmoid() * self.max_scale
        weights = self.qk_weights(x).view(B, H, W, self.groups, self.kernels)
        out = DCNFunction.apply(v, deformables, weights)
        out = out.view(B, H, W, -1)
        return self.out(out)


class FlowDCNBlock(nn.Module):
    def __init__(self, hidden_size, groups, kernels=9, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MultiScaleDCN(hidden_size, groups=groups, channels=hidden_size // groups, kernels=kernels)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa[:, None, None] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, None] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FlowDCN(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, num_groups=12,
                 hidden_size=1152, num_blocks=18, num_classes=1000,
                 learn_sigma=True, class_dropout_prob=0.0):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size, class_dropout_prob)
        self.label_dim = num_classes
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.blocks = nn.ModuleList([
            FlowDCNBlock(hidden_size, num_groups, kernels=9) for _ in range(num_blocks)
        ])
        self.initialize_weights()

    @property
    def img_channels(self):
        return 4

    @property
    def img_resolution(self):
        return 32

    def initialize_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, h, w):
        c = self.out_channels
        p = self.x_embedder.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        batch_size = x.shape[0]
        x = self.x_embedder(x)
        height, width = x.shape[2:]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        B, L, C = x.shape
        x = x.view(B, height, width, C)
        for block in self.blocks:
            x = block(x, c)
        x = x.view(B, L, C)
        x = self.final_layer(x, c)
        x = self.unpatchify(x, height, width)
        if self.learn_sigma:
            x, _ = torch.split(x, self.out_channels // 2, dim=1)
        return x
