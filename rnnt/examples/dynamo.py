import torch
import torch.fx

from functorch.compile import aot_module

from .. import model
from ..ref import load_weights
from .. import dataset


class PerfRecorder(torch.fx.Interpreter):
    def run_node(self, n : torch.fx.Node):
        print(f'Node: {n.target}')
        return super().run_node(n)

def print_compile_fn(gm : torch.fx.GraphModule, args):
    def wrapper(*args, **kwargs):
        out = PerfRecorder(gm).run(*args, **kwargs)
        print()
        return out

    return wrapper

def aot_compile_fn(gm : torch.fx.GraphModule, args):
    return aot_module(gm, print_compile_fn)

rnnt = model.Rnnt()
rnnt.eval()
rnnt.load_from_file('/research/data/mlmodels/npz/rnnt.npz')

dataset = dataset.Librespeech('./rnnt/librespeech-min/librespeech-min.json')

x = torch.Tensor(dataset[0].audio.samples).unsqueeze_(0)
l = torch.LongTensor([dataset[0].audio.num_samples])

compiled = torch.compile(rnnt, backend=aot_compile_fn)
# aot_rnnt = aot_module(rnnt, print_compile_fn, dynamic=True)

# rnnt.encoder = aot_module(rnnt.encoder, print_compile_fn)
# rnnt.prediction = aot_module(rnnt.prediction, print_compile_fn)
# rnnt.joint = aot_module(rnnt.joint, print_compile_fn)

compiled(x, l)
