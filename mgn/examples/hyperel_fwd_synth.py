import torch
import torch.utils.data

from .. import graphnet as GNN
from .. import hyperel

ds = hyperel.HyperElasticitySyntheticData(10, 10, 10, 1024)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda b: \
        GNN.collate_common(b, hyperel.HyperElasticitySampleBatch)
)

batch = next(iter(dl))

net = hyperel.HyperElasticityModel(
    hyperel.HyperElasticityModel.default_config_3d)

net.forward(batch)


