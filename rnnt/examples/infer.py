import torch
from .. import model
from ..ref import load_weights
from .. import dataset

rnnt = model.Rnnt()
rnnt.eval()
rnnt.load_from_file('/research/data/mlmodels/npz/rnnt.npz')

dataset = dataset.Librespeech('./rnnt/librespeech-min/librespeech-min.json')

x = torch.Tensor(dataset[0].audio.samples).unsqueeze_(0)
l = torch.LongTensor([dataset[0].audio.num_samples])

out = rnnt(x, l)

print(out)
