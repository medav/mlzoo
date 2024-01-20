import torch
import numpy as np
from dataclasses import dataclass, field

def exclusive_cumprod(x : torch.Tensor):
    return torch.cat([
        torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
        torch.cumprod(x, dim=-1)[..., :-1]],
    dim=-1)


class NerfEmbedding(torch.nn.Module):
    @dataclass
    class Params:
        identity : bool = True
        include_input : bool = True
        input_dims : int = 3
        max_freq_log2 : int = 9
        num_freqs : int = 10
        log_sampling : bool = True
        periodic_fns : list[str] = field(default_factory=lambda: ['sin', 'cos'])

    def __init__(self, params : Params):
        super().__init__()
        self.params = params


        if params.log_sampling:
            freq_bands = 2.0 ** np.linspace(
                0.0, params.max_freq_log2, params.num_freqs)
        else:
            freq_bands = np.linspace(
                2.0 ** 0.0, 2.0 ** params.max_freq_log2, params.num_freqs)

        self.freq_bands = freq_bands

        # TODO: Clean this up
        if params.identity:
            self.out_dim = params.input_dims
        else:
            self.out_dim = (
                len(freq_bands) * len(params.periodic_fns) +
                (1 if params.include_input else 0)
            ) * params.input_dims


    def forward(self, x : torch.Tensor):
        if self.params.identity: return x
        else:
            out = []
            if self.params.include_input: out.append(x)

            for freq in self.freq_bands:
                for fn in self.params.periodic_fns:
                    out.append({
                        'sin': torch.sin,
                        'cos': torch.cos
                    }[fn](freq * x))

            return torch.cat(out, dim=-1)



class Nerf(torch.nn.Module):
    @dataclass
    class InterpretedResult:
        rgb_map : torch.Tensor
        disp_map : torch.Tensor
        acc_map : torch.Tensor
        weights : torch.Tensor
        depth_map : torch.Tensor

    default_x_enc_params = NerfEmbedding.Params(
        identity=False,
        include_input=True,
        input_dims=3,
        max_freq_log2=9,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=['sin', 'cos'])

    default_d_enc_params = NerfEmbedding.Params(
        identity=False,
        include_input=True,
        input_dims=3,
        max_freq_log2=3,
        num_freqs=4,
        log_sampling=True,
        periodic_fns=['sin', 'cos'])

    def __init__(
        self,
        x_enc_params : NerfEmbedding.Params = default_x_enc_params,
        d_enc_params : NerfEmbedding.Params = default_d_enc_params,
        out_ch : int = 4,          # Num output channels
        hidden_dim : int = 256,    # Num features for hidden layers
        num_layers : int = 8,      # Num hidden layers,
        skip : int = 4,            # Which layer has skip connection
        use_viewdirs : bool = True # Use viewdirs as input
    ):
        super().__init__()

        preskip_layers = []
        postskip_layers = []

        self.x_enc = NerfEmbedding(x_enc_params)
        self.d_enc = NerfEmbedding(d_enc_params)

        in_features = self.x_enc.out_dim

        for i in range(num_layers):
            if i <= skip:
                preskip_layers.append(torch.nn.Linear(in_features, hidden_dim))
                preskip_layers.append(torch.nn.ReLU())
            else:
                postskip_layers.append(torch.nn.Linear(in_features, hidden_dim))
                postskip_layers.append(torch.nn.ReLU())

            if i != skip: in_features = hidden_dim
            else: in_features = in_features + self.x_enc.out_dim

        self.preskip = torch.nn.Sequential(*preskip_layers)
        self.postskip = torch.nn.Sequential(*postskip_layers)

        self.use_viewdirs = use_viewdirs

        if use_viewdirs:
            self.alpha = torch.nn.Linear(hidden_dim, 1)
            self.bottleneck = torch.nn.Linear(hidden_dim, 256)

            self.rgb = torch.nn.Sequential(*[
                torch.nn.Linear(256 + self.d_enc.out_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 3)
            ])

        else:
            self.decode = torch.nn.Linear(hidden_dim, out_ch)

    def forward(self, x : torch.Tensor, d : torch.Tensor = None):
        x_enc = self.x_enc(x)

        preskip_out = self.preskip(x_enc)
        resid = torch.cat([preskip_out, x_enc], dim=-1)
        encoded = self.postskip(resid)

        if self.use_viewdirs:
            assert d is not None
            d_enc = self.d_enc(d)

            alpha_out = self.alpha(encoded)
            bottleneck_out = self.bottleneck(encoded)
            rgb_out = self.rgb(torch.cat([bottleneck_out, d_enc], dim=-1))
            return torch.cat([rgb_out, alpha_out], dim=-1)

        else: return self.decode(encoded)

    def interpret(
        self,
        raw : torch.Tensor,    # [N, S, 4]
        z_vals : torch.Tensor, # [N, S]
        rays_d : torch.Tensor, # [N, 3]
        raw_noise_std : float = 0.0,
        white_bg : bool = False
    ):
        """Interpret raw network output.

        N.B. This is mostly a copy of `raw2outputs` from the original NeRF.
        """

        nrays = raw.shape[0]
        nsamp = raw.shape[1]

        # Compute 'distance' (in time) between each integration time along a
        # ray. The 'distance' from the last integration time is infinity.
        # dists : [N, S]
        dists = torch.cat([
            z_vals[..., 1:] - z_vals[..., :-1],
            torch.full((nrays, 1), 65504, device=raw.device, dtype=raw.dtype)
        ], axis=-1) * torch.norm(rays_d, dim=-1, keepdim=True)

        # rgb : [N, S, 3]
        rgb = torch.sigmoid(raw[..., :3])

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn((nrays, nsamp)) * raw_noise_std

        # alpha : [N, S]
        alpha = 1. - torch.exp(-torch.relu(raw[..., 3] + noise) * dists)

        # weights : [N, S]
        weights = alpha * exclusive_cumprod(1. - alpha + 1e-10)

        # Weighted color of each sample along each ray.
        # rgb_map : [N, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

        # Weighted depth of each sample along each ray.
        # depth_map : [N]
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        # disp_map : [N]
        disp_map = torch.reciprocal(
            torch.max(
                1e-10 * torch.ones_like(depth_map),
                depth_map / torch.norm(rays_d, dim=-1)))

        # Sum of weights along each ray.
        # acc_map : [N]
        acc_map = torch.sum(weights, dim=-1)

        if white_bg: rgb_map = rgb_map + (1 - acc_map[..., None])

        return Nerf.InterpretedResult(
            rgb_map, disp_map, acc_map, weights, depth_map)


    def render_rays(
        self,
        rays_o : torch.Tensor,
        rays_d : torch.Tensor,
        focal : float,
        coarse_samples : int,
        fine_samples : int = 0,
        near : int = 0,
        far : int = 1,
        lindisp : bool = False
    ):
        num_rays = rays_o.shape[0]
        t_vals = torch.linspace(
            0, 1, steps=coarse_samples, device=rays_o.device, dtype=rays_o.dtype)

        if lindisp: z_vals = near * (1 - t_vals) + far * t_vals
        else: z_vals = 1 / (1 / near * (1 - t_vals) + 1 / far * t_vals)

        z_vals = z_vals.broadcast_to([num_rays, coarse_samples])
        xs = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        if self.use_viewdirs:
            viewdirs = torch.nn.functional.normalize(rays_d, dim=-1) \
                .unsqueeze(-2) \
                .broadcast_to(xs.shape)
        else: viewdirs = None

        res = self.interpret(self.forward(xs, viewdirs), z_vals, rays_d)

        return res





