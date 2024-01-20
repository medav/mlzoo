import torch
import torch.fx
import torchvision

from functorch.compile import aot_module

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

batch_size = 64
dtype = torch.float16
dev = torch.device('cuda:0')

net = torchvision.models.resnet50(pretrained=True).to(dtype).to(dev)
x = torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)


compiled = torch.compile(net, backend=print_compile_fn)

# compiled(x)

aot_fn = aot_module(
    net, fw_compiler=print_compile_fn)


aot_fn(x)
