import torch
import pandas as pd
import numpy as np
from . import model


class CovertypeDataset(torch.utils.data.Dataset):
    feature_cols = [
        model.NumericColumn(0, 'Elevation'),
        model.NumericColumn(1, 'Aspect'),
        model.NumericColumn(2, 'Slope'),
        model.NumericColumn(3, 'Horizontal_Distance_To_Hydrology'),
        model.NumericColumn(4, 'Vertical_Distance_To_Hydrology'),
        model.NumericColumn(5, 'Horizontal_Distance_To_Roadways'),
        model.NumericColumn(6, 'Hillshade_9am'),
        model.NumericColumn(7, 'Hillshade_Noon'),
        model.NumericColumn(8, 'Hillshade_3pm'),
        model.NumericColumn(9, 'Horizontal_Distance_To_Fire_Points')
    ] + [
        model.BoolColumn(10 + i, f'Wilderness_Area{i + 1}')
        for i in range(4)
    ] + [
        model.BoolColumn(14 + i, f'Soil_Type{i + 1}')
        for i in range(40)
    ]

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

if __name__ == '__main__':
    ds = CovertypeDataset('tabnet/ref/data/train_covertype.csv')
    print(ds[0:20])
