import torch
import torch.utils.data

from .. import model
from .. import dataset

config = model.TabNet.covertype_config

net = model.TabNet(config).cuda()

ds_train = dataset.CovertypeDataset('tabnet/ref/data/train_covertype.csv')
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=1024, shuffle=True)

ds_val = dataset.CovertypeDataset('tabnet/ref/data/val_covertype.csv')

optimizer = torch.optim.Adam(net.parameters(), lr=0.02, betas=(0.9, 0.999))


def val():
    print('Testing...')
    with torch.no_grad():
        cols, labels = ds_val[:]
        cols = [c.cuda() for c in cols]
        labels = labels.cuda()
        acc = torch.mean((net.classify(cols) == labels).float()).item()
        print(f'Val acc: {acc}')

for i in range(10):
    val()

    print(f'Epoch {i}:')
    for cols, labels in dl_train:
        cols = [c.cuda() for c in cols]
        labels = labels.cuda()

        optimizer.zero_grad()
        loss = net.loss(cols, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())

val()
