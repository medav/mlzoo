from dataclasses import dataclass
import enum
from typing import Optional
import torch
import numpy as np
from . import unsorted_segsum

def make_torch_param(data): return torch.nn.Parameter(torch.tensor(data))
def make_torch_buffer(data): return torch.tensor(data)

def cells_to_edges(cells : torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    cells: int32[M, D]
    :ret: int32[E], int32[E]
    """

    if cells.shape[1] == 3:
        # Triangles

        raw_edges = torch.cat([
            cells[:, 0:2],
            cells[:, 1:3],
            torch.stack([cells[:, 2], cells[:, 0]], dim=1)
        ], dim=0)

    elif cells.shape[1] == 4:
        # Tetrahedrons

        raw_edges = torch.cat([
            cells[:, 0:2],
            cells[:, 1:3],
            cells[:, 2:4],
            torch.stack([cells[:, 0], cells[:, 2]], dim=1),
            torch.stack([cells[:, 0], cells[:, 3]], dim=1),
            torch.stack([cells[:, 1], cells[:, 3]], dim=1)
        ], dim=0)

    else: raise NotImplementedError('Unknown cell type')

    srcs = raw_edges.max(dim=1).values
    dsts = raw_edges.min(dim=1).values

    edges = torch.stack([srcs, dsts], dim=1)
    unique_edges = edges.unique(dim=0, sorted=False)
    srcs, dsts = unique_edges[:, 0], unique_edges[:, 1]

    srcs, dsts = torch.cat([srcs, dsts], dim=0), torch.cat([dsts, srcs], dim=0)

    return srcs, dsts


@dataclass
class GraphNetSample:
    cells : torch.Tensor
    node_type : torch.Tensor
    mesh_pos : torch.Tensor

    def todev(self, dev):
        def _todev(x):
            return x.to(dev)

        fields = self.__dataclass_fields__.keys()
        for field in fields:
            setattr(self, field, _todev(getattr(self, field)))

        return self

    def asdtype(self, dtype):
        def _asdtype(x):
            if torch.is_floating_point(x): return x.to(dtype)
            else: return x

        fields = self.__dataclass_fields__.keys()
        for field in fields:
            setattr(self, field, _asdtype(getattr(self, field)))

        return self

@dataclass
class GraphNetSampleBatch(GraphNetSample):
    node_offs : torch.Tensor

def collate_common(batch : list[GraphNetSample], ty):
    custom_field_names = set(batch[0].__dataclass_fields__.keys()) - \
        set(GraphNetSample.__dataclass_fields__.keys())

    node_offs = torch.LongTensor([
        0 if i == 0 else batch[i - 1].node_type.size(0)
        for i in range(len(batch))
    ]).cumsum(dim=0)

    cells = torch.cat([
        b.cells + node_offs[i]
        for i, b in enumerate(batch)
    ], dim=0)

    custom_fields = {
        k: torch.cat([getattr(b, k) for b in batch], dim=0)
        for k in custom_field_names
    }

    return ty(
        node_offs=node_offs,
        cells=cells,
        node_type=torch.cat([b.node_type for b in batch], dim=0),
        mesh_pos=torch.cat([b.mesh_pos for b in batch], dim=0),
        **custom_fields
    )

def load_npz_common(path : str, type) -> "type":
    np_data = np.load(path)

    return type(**{
        k: torch.from_numpy(v)
        for k, v in np_data.items()
        if k in type.__dataclass_fields__.keys()
    })

@dataclass
class EdgeSet:
    features : torch.Tensor
    senders : torch.Tensor
    receivers : torch.Tensor
    offsets : Optional[torch.Tensor] = None

    def sort_(self, num_nodes):
        idxs = torch.argsort(self.receivers)
        self.features = self.features[idxs]
        self.senders = self.senders[idxs]
        self.receivers = self.receivers[idxs]

@dataclass
class MultiGraph:
    node_features : torch.Tensor
    edge_sets : list[EdgeSet]


class InvertableNorm(torch.nn.Module):
    def __init__(
        self,
        shape : tuple[int],
        eps: float = 1e-8,
        max_accumulations : int = 10**6
    ) -> None:
        super().__init__()
        self.shape = shape
        self.register_buffer('eps', torch.Tensor([eps]))
        self.eps : torch.Tensor
        self.max_accumulations = max_accumulations
        self.frozen = False

        self.register_buffer('running_sum', torch.zeros(shape))
        self.register_buffer('running_sum_sq', torch.zeros(shape))
        self.running_sum: torch.Tensor
        self.running_sum_sq: torch.Tensor

        self.register_buffer('num_accum', torch.tensor(0, dtype=torch.long))
        self.num_accum: torch.Tensor

        self.register_buffer('accum_count', torch.tensor(0, dtype=torch.long))
        self.accum_count: torch.Tensor

    @property
    def stats(self) -> torch.Tensor:
        num_accum = max(self.num_accum.item(), 1)

        mean = self.running_sum / num_accum
        std = torch.max(
            torch.sqrt(self.running_sum_sq / num_accum - mean**2),
            self.eps)

        return mean, std

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        assert x.shape[-len(self.shape):] == self.shape, f'Expected shape {self.shape}, got {x.shape}'
        n_batch_dims = x.ndim - len(self.shape)
        batch_dims = tuple(i for i in range(n_batch_dims))

        if self.accum_count.item() < self.max_accumulations and not self.frozen:
            self.running_sum += x.sum(dim=batch_dims)
            self.running_sum_sq += x.pow(2).sum(dim=batch_dims)
            self.num_accum += np.prod(list(x.shape[i] for i in batch_dims))
            self.accum_count += 1

        mean, std = self.stats
        return (x - mean) / std

    def inverse(self, x : torch.Tensor) -> torch.Tensor:
        mean, std = self.stats
        return x * std + mean


class Mlp(torch.nn.Module):
    def __init__(self, input_size : int, widths : list[int], layernorm=True):
        super().__init__()
        widths = [input_size] + widths
        modules = []
        for i in range(len(widths) - 1):
            if i < len(widths) - 2:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1]), torch.nn.ReLU()))
            else:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1])))

        if layernorm: modules.append(torch.nn.LayerNorm(widths[-1]))
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x): return self.model(x)


class GraphNetBlock(torch.nn.Module):
    @dataclass
    class Config:
        node_feature_dim : int
        edge_feature_dim : int
        num_edge_sets : int
        mlp_widths : list[int]

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        self.node_mlp = Mlp(
            config.node_feature_dim + \
                config.num_edge_sets * config.edge_feature_dim,
            config.mlp_widths,
            layernorm=True)

        self.edge_mlps = torch.nn.ModuleList([
            Mlp(
                2 * config.node_feature_dim + config.edge_feature_dim,
                config.mlp_widths,
                layernorm=True)
            for _ in range(config.num_edge_sets)
        ])

    def _update_node_features(
        self,
        node_features : torch.Tensor,
        edge_sets : list[EdgeSet]
    ) -> torch.Tensor:
        num_nodes = node_features.size(0)
        features = [node_features]

        for edge_set in edge_sets:
            features.append(unsorted_segsum.unsorted_segment_sum(
                edge_set.features, edge_set.receivers, num_nodes))

        return self.node_mlp(torch.cat(features, dim=-1))

    def _update_edge_features(
        self,
        i : int,
        node_features : torch.Tensor,
        edge_set : EdgeSet
    ) -> torch.Tensor:
        srcs = node_features[edge_set.senders, :]
        dsts = node_features[edge_set.receivers, :]
        edge_features = edge_set.features
        return self.edge_mlps[i](torch.cat([srcs, dsts, edge_features], dim=-1))

    def forward(self, graph : MultiGraph) -> MultiGraph:
        node_features = graph.node_features
        edge_sets = graph.edge_sets

        assert len(edge_sets) == self.config.num_edge_sets

        new_edge_sets = [
            EdgeSet(
                features=self._update_edge_features(i, node_features, edge_set),
                senders=edge_set.senders,
                receivers=edge_set.receivers,
                offsets=edge_set.offsets
            )
            for i, edge_set in enumerate(edge_sets)
        ]

        new_node_features = \
            self._update_node_features(node_features, new_edge_sets)

        for ei in range(self.config.num_edge_sets):
            new_edge_sets[ei].features = \
                new_edge_sets[ei].features + edge_sets[ei].features

        return MultiGraph(new_node_features + node_features, new_edge_sets)

class GraphNetEncoder(torch.nn.Module):
    @dataclass
    class Config:
        node_input_dim : int
        edge_input_dims : list[int]
        latent_size : int
        num_edge_sets : int
        num_layers : int

    def __init__(self, config : Config):
        super().__init__()
        mlp_widths = \
            [config.latent_size] * config.num_layers + [config.latent_size]

        self.node_mlp = Mlp(config.node_input_dim, mlp_widths, layernorm=True)
        self.edge_mlps = torch.nn.ModuleList([
            Mlp(config.edge_input_dims[i], mlp_widths, layernorm=True)
            for i in range(config.num_edge_sets)
        ])

    def forward(self, graph : MultiGraph) -> MultiGraph:
        return MultiGraph(
            node_features=self.node_mlp(graph.node_features),
            edge_sets=[
                EdgeSet(
                    features=self.edge_mlps[i](edge_set.features),
                    senders=edge_set.senders,
                    receivers=edge_set.receivers
                )
                for i, edge_set in enumerate(graph.edge_sets)
            ])

class GraphNetDecoder(torch.nn.Module):
    @dataclass
    class Config:
        latent_size : int
        output_size : int
        num_layers : int

    def __init__(self, config : Config):
        super().__init__()
        mlp_widths = \
            [config.latent_size] * config.num_layers + [config.output_size]

        self.node_mlp = Mlp(config.latent_size, mlp_widths, layernorm=False)

    def forward(self, graph : MultiGraph) -> torch.Tensor:
        return self.node_mlp(graph.node_features)

class GraphNetModel(torch.nn.Module):
    @dataclass
    class Config:
        node_input_dim : int
        edge_input_dims : list[int]
        output_dim : int
        latent_size : int
        num_edge_sets : int
        num_mlp_layers : int
        num_mp_steps : int

    def __init__(self, config : Config):
        super().__init__()

        self.encoder = GraphNetEncoder(
            GraphNetEncoder.Config(
                node_input_dim=config.node_input_dim,
                edge_input_dims=config.edge_input_dims,
                latent_size=config.latent_size,
                num_edge_sets=config.num_edge_sets,
                num_layers=config.num_mlp_layers
            )
        )

        self.decoder = GraphNetDecoder(
            GraphNetDecoder.Config(
                latent_size=config.latent_size,
                output_size=config.output_dim,
                num_layers=config.num_mlp_layers
            )
        )

        block_config = GraphNetBlock.Config(
            node_feature_dim=config.latent_size,
            edge_feature_dim=config.latent_size,
            num_edge_sets=config.num_edge_sets,
            mlp_widths=[config.latent_size] * config.num_mlp_layers + \
                [config.latent_size]
        )

        self.blocks = torch.nn.ModuleList([
            GraphNetBlock(block_config)
            for _ in range(config.num_mp_steps)
        ])

    def forward(self, graph : MultiGraph) -> torch.Tensor:
        graph = self.encoder(graph)

        for block in self.blocks:
            graph = block(graph)

        return self.decoder(graph)



