import torch
from .. import model


batch_size = 2048
idxs_per_sample = 64

net = model.Dlrm(model.dlrm_mlperf_no_emb_config).cuda()

dense_x = torch.randn(batch_size, 13, dtype=torch.float32).cuda()
sparse_idxs = [
    torch.randint(0, 1, (batch_size * idxs_per_sample,), dtype=torch.long).cuda()
    for _ in range(26)
]

sparse_offs = [
    torch.arange(batch_size, dtype=torch.long).cuda() * idxs_per_sample
    for _ in range(26)
]

out = net(dense_x, sparse_idxs, sparse_offs)

