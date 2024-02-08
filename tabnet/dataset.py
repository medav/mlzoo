import torch
import pandas as pd
import numpy as np
from . import model


class CovertypeDataset(torch.utils.data.Dataset):
    feature_cols = model.TabNet.covertype_cols
    label_col = model.IntColumn(54, 'CoverType')

    def __init__(self, filename):
        self.df = pd.read_csv(filename, header=None)

        self.features = [
            col.preprocess(self.df[col.idx].to_numpy()).astype(col.dtype)
            for col in self.feature_cols
        ]

        self.label = self.label_col.preprocess(
            self.df[self.label_col.idx].to_numpy()) \
                .astype(self.label_col.dtype)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return \
            [torch.tensor(self.features[i][idx]) for i in range(len(self.features))], \
            torch.tensor(self.label[idx])

class CovertypeSyntheticDataset(torch.utils.data.Dataset):
    feature_cols = model.TabNet.covertype_cols
    label_col = model.IntColumn(54, 'CoverType')

    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.features = [
            col.make_synthetic(num_samples)
            for col in self.feature_cols
        ]

        self.label = self.label_col.make_synthetic(num_samples)

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        return \
            [torch.tensor(self.features[i][idx]) for i in range(len(self.features))], \
            torch.tensor(self.label[idx])

if __name__ == '__main__':
    ds = CovertypeDataset('tabnet/ref/data/train_covertype.csv')
    print(ds[0:20])
