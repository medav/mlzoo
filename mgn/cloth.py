import enum
import torch
import json
import os
import numpy as np
from dataclasses import dataclass

from . import graphnet as GNN
from . import meshgen

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

@dataclass
class ClothSample(GNN.GraphNetSample):
    world_pos : torch.Tensor
    prev_world_pos : torch.Tensor
    target_world_pos : torch.Tensor

@dataclass
class ClothSampleBatch(GNN.GraphNetSampleBatch):
    world_pos : torch.Tensor
    prev_world_pos : torch.Tensor
    target_world_pos : torch.Tensor

class ClothModel(torch.nn.Module):
    @dataclass
    class Config:
        mesh_space_dim : int
        world_space_dim : int
        latent_size : int
        num_mlp_layers : int
        num_mp_steps : int

        @property
        def num_edge_sets(self): return 1

        @property
        def n_edge_f(self):
            return self.mesh_space_dim + self.world_space_dim + 2

        @property
        def input_dim(self): return self.world_space_dim + NodeType.SIZE

        @property
        def output_dim(self): return self.world_space_dim

        @property
        def graphnet_config(self):
            return GNN.GraphNetModel.Config(
                node_input_dim=self.input_dim,
                edge_input_dims=[self.n_edge_f],
                output_dim=self.output_dim,
                latent_size=self.latent_size,
                num_edge_sets=self.num_edge_sets,
                num_mlp_layers=self.num_mlp_layers,
                num_mp_steps=self.num_mp_steps
            )

    default_config = Config(
        mesh_space_dim=2,
        world_space_dim=3,
        latent_size=128,
        num_mlp_layers=2,
        num_mp_steps=15
    )

    def __init__(self, config : Config = default_config):
        super().__init__()
        self.graph_net = GNN.GraphNetModel(config.graphnet_config)
        self.out_norm = GNN.InvertableNorm((config.output_dim,))
        self.node_norm = GNN.InvertableNorm((config.input_dim,))
        self.edge_norm = GNN.InvertableNorm((config.n_edge_f,))

    def forward(self, x : ClothSampleBatch, unnorm : bool = True) -> torch.Tensor:
        """Predicts Delta V"""

        node_type_oh = \
            torch.nn.functional.one_hot(x.node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        velocity = x.world_pos - x.prev_world_pos

        node_features = torch.cat([velocity, node_type_oh], dim=-1)

        srcs, dsts = GNN.cells_to_edges(x.cells)
        rel_mesh_pos = x.mesh_pos[srcs, :] - x.mesh_pos[dsts, :]
        rel_world_pos = x.world_pos[srcs, :] - x.world_pos[dsts, :]

        edge_features = torch.cat([
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
            rel_world_pos,
            torch.norm(rel_world_pos, dim=-1, keepdim=True)
        ], dim=-1)

        graph = GNN.MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[GNN.EdgeSet(self.edge_norm(edge_features), srcs, dsts)]
        )

        net_out = self.graph_net(graph)

        if unnorm: return self.out_norm.inverse(net_out)
        else: return net_out

    def loss(self, x : ClothSampleBatch) -> torch.Tensor:
        pred = self.forward(x, unnorm=False)

        with torch.no_grad():
            target_accel = x.target_world_pos - 2 * x.world_pos + x.prev_world_pos
            target_accel_norm = self.out_norm(target_accel)

        residuals = (target_accel_norm - pred).sum(dim=-1)
        mask = (x.node_type == NodeType.NORMAL).squeeze()
        return residuals[mask].pow(2).mean()

class ClothData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'meta.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = np.load(os.path.join(self.path, fname))
            total_nodes += data['mesh_pos'].shape[1] * (num_steps - 1)
            total_samples += num_steps - 1

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 1): return fname, sample_id
            else: sample_id -= (num_steps - 1)
        raise IndexError()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = np.load(os.path.join(self.path, fname))

        return ClothSample(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]).squeeze(),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )

class ClothSyntheticData(ClothData):
    def __init__(self, nx, ny, num_samples):
        self.num_samples = num_samples

        cells, pos = meshgen.gen_2d_tri_mesh(nx, ny)
        world_pos = torch.cat([pos, torch.zeros(pos.shape[0], 1)], dim=-1)

        self.sample = ClothSample(
            cells=cells,
            node_type=torch.zeros(pos.shape[0], dtype=torch.int64),
            mesh_pos=pos,
            world_pos=world_pos * 2.0,
            prev_world_pos=world_pos - 0.5,
            target_world_pos=world_pos * 0.5
        )

    def __len__(self): return self.num_samples
    def __getitem__(self, idx : int) -> dict: return self.sample

sample_type = ClothSample
batch_type = ClothSampleBatch
model_type = ClothModel
dataset_type = ClothData
