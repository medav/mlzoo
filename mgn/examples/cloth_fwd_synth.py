import torch
import torch.utils.data

from .. import graphnet as GNN
from .. import cloth

ds = cloth.ClothSyntheticData(10, 10, 1024)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda b: \
        GNN.collate_common(b, cloth.ClothSampleBatch)
)

batch = next(iter(dl))

net = cloth.ClothModel(cloth.ClothModel.default_config)

net.forward(batch)


