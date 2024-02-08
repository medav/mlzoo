import torch
import torch.utils.data

from .. import model
from .. import dataset

net = model.TabNet(model.TabNet.covertype_config).cuda()

ds = dataset.CovertypeSyntheticDataset(1024)
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)

batch = next(iter(dl))

cols, labels = batch
cols = [c.cuda() for c in cols]
labels = labels.cuda()

print(net(cols))

