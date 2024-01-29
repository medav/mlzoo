import torch
from dataclasses import dataclass, field

class DlrmMlp(torch.nn.Module):
    def __init__(self, widths : list[int], sigmoid_i=None):
        super().__init__()
        modules = []
        for i in range(len(widths) - 1):
            modules.append(torch.nn.Linear(widths[i], widths[i + 1]))

            if i == sigmoid_i: modules.append(torch.nn.Sigmoid())
            else: modules.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x): return self.model(x)

class Dlrm(torch.nn.Module):
    @dataclass
    class Config:
        mlp_bot_n : list[int] = field(default_factory=list)
        mlp_top_n : list[int] = field(default_factory=list)
        emb_size : list[int] = field(default_factory=list)
        emb_dim : int = 128
        interaction : str = 'dot'
        interact_self : bool = False

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        self.bot_mlp = DlrmMlp(config.mlp_bot_n)
        self.top_mlp = DlrmMlp(config.mlp_top_n)

        assert config.interaction == 'dot', \
            'Only dot product interaction is supported'

        self.embs = torch.nn.ModuleList([
            torch.nn.EmbeddingBag(n, config.emb_dim, mode='sum', sparse=True)
            for n in config.emb_size
        ])

        ni = len(config.emb_size) + 1
        nj = len(config.emb_size) + 1

        off = 1 if config.interact_self else 0

        self.register_buffer(
            'li',
            torch.tensor([i for i in range(ni) for j in range(i + off)]),
            persistent=False)

        self.li : torch.Tensor

        self.register_buffer(
            'lj',
            torch.tensor([j for i in range(nj) for j in range(i + off)]),
            persistent=False)

        self.lj : torch.Tensor

    def forward(
        self,
        dense_x : torch.Tensor, # [B, D]
        sparse_idxs : list[torch.Tensor], # [B x N = avg # sparse features per sample]
        sparse_offs : list[torch.Tensor] # [B]
    ):
        bot_mlp_out = self.bot_mlp(dense_x)
        B, D = bot_mlp_out.shape

        features = torch.cat([bot_mlp_out] + [
            emb(s_idx, s_off)
            for emb, s_idx, s_off in zip(self.embs, sparse_idxs, sparse_offs)
        ], dim=1).view(B, -1, D)

        interact_out = \
            torch.bmm(features, features.transpose(1, 2))[:, self.li, self.lj]

        top_mlp_in = torch.cat([bot_mlp_out, interact_out], dim=1)

        return self.top_mlp(top_mlp_in)


dlrm_mlperf_no_emb_config = Dlrm.Config(
    mlp_bot_n=[13, 512, 256, 128],
    mlp_top_n=[479, 1024, 1024, 512, 256, 1],
    emb_size=[1] * 26,
    emb_dim=128,
    interaction='dot',
    interact_self=False
)

