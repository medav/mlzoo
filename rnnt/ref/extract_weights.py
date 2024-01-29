import argparse
from . import pytorch_SUT
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str, required=True)

args = parser.parse_args()

rnnt = pytorch_SUT.PytorchSUT(
    './rnnt/ref/configs/rnnt.toml', args.checkpoint_dir)

model = rnnt.model

print(model)

for n, p in model.named_parameters(): print(n, p.shape)
for n, p in model.named_buffers(): print(n, p.shape)

params = {
    name: p.detach().numpy()
    for name, p in model.named_parameters()
}

params.update({
    name: b.detach().numpy()
    for name, b in model.named_buffers()
})

np.savez('rnnt.npz', **params)
