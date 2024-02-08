import torch
import torch.utils.data

import torch.fx
from functorch.compile import aot_module
from functorch.compile import make_boxed_func

from .. import graphnet as GNN
from .. import hyperel
from .. import meshgen

def trace_compile_fn(gm : torch.fx.GraphModule, args):
    gm.graph.print_tabular()
    print()
    return make_boxed_func(gm.forward)

def aot_compile_fn(gm : torch.fx.GraphModule, args):
    return aot_module(gm, trace_compile_fn)


ds = hyperel.HyperElasticitySyntheticData(10, 10, 10, 10, 10, 1, 1024)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda b: \
        GNN.collate_common(b, hyperel.HyperElasticitySampleBatch)
)

batch = next(iter(dl)).todev('cuda:0')

net = hyperel.HyperElasticityModel(
    hyperel.HyperElasticityModel.default_config_3d).to('cuda:0')

compiled = torch.compile(net, backend=aot_compile_fn)
compiled(batch)


