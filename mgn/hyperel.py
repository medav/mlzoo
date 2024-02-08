# The following is based on:
# https://github.com/deepmind/deepmind-research/compare/master...isabellahuang:deepmind-research-tf1:tf1
#
# Subject to Apache v2 license (see original source for details)

import torch
import random
import enum
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
class HyperElasticitySample(GNN.GraphNetSample):
    world_pos : torch.Tensor
    target_world_pos : torch.Tensor
    stress : torch.Tensor

@dataclass
class HyperElasticitySampleBatch(GNN.GraphNetSampleBatch):
    world_pos : torch.Tensor
    target_world_pos : torch.Tensor
    stress : torch.Tensor


def arange2d(m, n, dev):
    return torch.stack([
        torch.arange(m, device=dev).reshape(-1, 1).repeat(1, n),
        torch.arange(n, device=dev).reshape(1, -1).repeat(m, 1)
    ], dim=2)

def squared_dist(A : torch.Tensor, B : torch.Tensor):
    row_norms_A = A.pow(2).sum(dim=1).reshape(-1, 1) # N, 1
    row_norms_B = B.pow(2).sum(dim=1).reshape(1, -1) # 1, N
    return row_norms_A - 2 * (A @ B.t()) + row_norms_B

def construct_world_edges(
    node_offs : torch.LongTensor,
    world_pos : torch.Tensor,
    node_type : torch.Tensor,
    thresh : float = 0.03
) -> torch.Tensor:
    dev = world_pos.device

    srcss = []
    dstss = []

    for bi in range(len(node_offs)):
        off = node_offs[bi]
        nn = (node_offs[bi + 1] if bi < len(node_offs) - 1 else world_pos.shape[0]) - off

        b_node_type = node_type[off:off + nn]
        b_world_pos = world_pos[off:off + nn, :]

        # print(f'Batch {bi} ({off}-{off + nn}): {b_node_type.shape} {b_world_pos.shape}')

        deformable = b_node_type != NodeType.OBSTACLE
        deformable_idx = torch.arange(b_node_type.shape[0], device=dev)[deformable]

        actuator = b_node_type == NodeType.OBSTACLE
        actuator_idx = torch.arange(b_node_type.shape[0], device=dev)[actuator]

        b_actuator_pos = b_world_pos[actuator].to(torch.float64) # M, D
        b_deformable_pos = b_world_pos[deformable].to(torch.float64) # N, D

        b_dists = squared_dist(b_actuator_pos, b_deformable_pos) # M, N
        M, N = b_dists.shape

        idxs = arange2d(M, N, dev)
        rel_close_pair_idx = idxs[b_dists < (thresh ** 2)]

        srcs = actuator_idx[rel_close_pair_idx[:, 0]] + off
        dsts = deformable_idx[rel_close_pair_idx[:, 1]] + off

        srcss.append(srcs)
        srcss.append(dsts)
        dstss.append(dsts)
        dstss.append(srcs)

    return torch.cat(srcss, dim=0), torch.cat(dstss, dim=0)

class HyperElasticityModel(torch.nn.Module):
    @dataclass
    class Config:
        space_dim : int
        latent_size : int
        num_mlp_layers : int
        num_mp_steps : int

        @property
        def num_edge_sets(self): return 2

        @property
        def input_dim(self): return self.space_dim + NodeType.SIZE

        @property
        def n_mesh_edge_f(self): return 2 * (self.space_dim + 1)

        @property
        def n_world_edge_f(self): return self.space_dim + 1

        @property
        def output_dim(self): return self.space_dim

        @property
        def graphnet_config(self):
            return GNN.GraphNetModel.Config(
                node_input_dim=self.input_dim,
                edge_input_dims=[self.n_mesh_edge_f, self.n_world_edge_f],
                output_dim=self.output_dim,
                latent_size=self.latent_size,
                num_edge_sets=self.num_edge_sets,
                num_mlp_layers=self.num_mlp_layers,
                num_mp_steps=self.num_mp_steps
            )

    default_config_3d = Config(
        space_dim=3,
        latent_size=128,
        num_mlp_layers=2,
        num_mp_steps=15
    )

    default_config_2d = Config(
        space_dim=2,
        latent_size=128,
        num_mlp_layers=2,
        num_mp_steps=15
    )

    def __init__(self, config : Config = default_config_3d):
        super().__init__()
        self.graph_net = GNN.GraphNetModel(config.graphnet_config)
        self.out_norm = GNN.InvertableNorm((config.output_dim,))
        self.node_norm = GNN.InvertableNorm((config.input_dim,))
        self.mesh_edge_norm = GNN.InvertableNorm((config.n_mesh_edge_f,))
        self.world_edge_norm = GNN.InvertableNorm((config.n_world_edge_f,))

    def forward(
        self,
        x : HyperElasticitySampleBatch,
        unnorm : bool = True
    ) -> torch.Tensor:
        """Predicts Velocity"""

        #
        # Node Features
        #

        node_type_oh = \
            torch.nn.functional.one_hot(x.node_type, num_classes=NodeType.SIZE) \
                .squeeze()

        known_vel = x.target_world_pos - x.world_pos
        known_vel[x.node_type != NodeType.NORMAL, :] = 0.0
        node_features = torch.cat([known_vel, node_type_oh], dim=-1)

        #
        # Mesh Edge Features
        #

        srcs, dsts = GNN.cells_to_edges(x.cells)
        rel_mesh_pos = x.mesh_pos[srcs, :] - x.mesh_pos[dsts, :]
        rel_world_mesh_pos = x.world_pos[srcs, :] - x.world_pos[dsts, :]

        mesh_edge_features = torch.cat([
            rel_world_mesh_pos,
            torch.norm(rel_world_mesh_pos, dim=-1, keepdim=True),
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True)
        ], dim=-1)


        #
        # World Edge Features
        #

        wsrcs, wdsts = construct_world_edges(x.node_offs, x.world_pos, x.node_type)
        rel_world_pos = x.world_pos[wsrcs, :] - x.world_pos[wdsts, :]

        world_edge_features = torch.cat([
            rel_world_pos,
            torch.norm(rel_world_pos, dim=-1, keepdim=True)
        ], dim=-1)


        graph = GNN.MultiGraph(
            node_features=self.node_norm(node_features),
            edge_sets=[
                GNN.EdgeSet(self.mesh_edge_norm(mesh_edge_features), srcs, dsts),
                GNN.EdgeSet(self.world_edge_norm(world_edge_features), wsrcs, wdsts)
            ]
        )

        net_out = self.graph_net(graph)

        if unnorm: return self.out_norm.inverse(net_out)
        else: return net_out

    def loss(self, x : HyperElasticitySampleBatch) -> torch.Tensor:
        pred = self.forward(x, unnorm=False)

        with torch.no_grad():
            delta_x = x.target_world_pos - x.world_pos
            delta_x_norm = self.out_norm(delta_x)

        residuals = (delta_x_norm - pred).sum(dim=-1)
        mask = (x.node_type == NodeType.NORMAL).squeeze()
        return residuals[mask].pow(2).mean()


class HyperElasticityData(torch.utils.data.Dataset):
    # Train set has avg 1276 nodes/samp

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

        return HyperElasticitySample(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]).squeeze(),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            stress=torch.Tensor(data['stress'][sid, ...])
        )

class HyperElasticitySyntheticData(HyperElasticityData):
    def __init__(self, nx, ny, nz, num_samples):
        self.num_samples = num_samples

        cells, pos = meshgen.gen_3d_tet_mesh(nx, ny, nz)

        self.sample = HyperElasticitySample(
            cells=cells,
            node_type=torch.zeros(pos.shape[0], dtype=torch.int64),
            mesh_pos=pos,
            world_pos=pos * 2.0,
            target_world_pos=pos * 0.5,
            stress=torch.randn(pos.shape[0], dtype=torch.float32)
        )

    def __len__(self): return self.num_samples
    def __getitem__(self, idx : int) -> dict: return self.sample

sample_type = HyperElasticitySample
batch_type = HyperElasticitySampleBatch
model_type = HyperElasticityModel
dataset_type = HyperElasticityData
