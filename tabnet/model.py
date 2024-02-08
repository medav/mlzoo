import torch
import numpy as np
from dataclasses import dataclass
from hashlib import md5


@dataclass
class FeatureColumn:
    idx : int
    name : str

    @property
    def dtype(self): return np.float32

    def make_synthetic(self, num_samples : int) -> np.ndarray:
        return np.random.random(num_samples).astype(self.dtype)

    def preprocess(self, col : np.ndarray) -> np.ndarray: return col

    def make_encoder(self) -> torch.nn.Module:
        return torch.nn.Identity()


@dataclass
class NumericColumn(FeatureColumn): pass

@dataclass
class IntColumn(NumericColumn):
    @property
    def dtype(self): return np.int64

    def make_synthetic(self, num_samples : int) -> np.ndarray:
        return np.random.randint(0, 10, num_samples).astype(self.dtype)

@dataclass
class CategoricalColumn(FeatureColumn):
    num_buckets : int
    emb_dim : int = 1
    hash_fn : callable = \
        np.vectorize(
            lambda x: int.from_bytes(md5(str(x).encode()).digest(), 'little'))

    @property
    def dtype(self): return np.int64

    def make_synthetic(self, num_samples : int) -> np.ndarray:
        return np.random.randint(0, self.num_buckets, num_samples) \
            .astype(self.dtype)

    def preprocess(self, col : np.ndarray) -> np.ndarray:
        return self.hash_fn(col) % self.num_buckets

    def make_encoder(self) -> torch.nn.Module:
        return torch.nn.Embedding(self.num_buckets, self.emb_dim)

@dataclass
class BoolColumn(CategoricalColumn):
    num_buckets : int = 3


def glu(act):
    return act[:, :act.size(1) // 2] * torch.sigmoid(act[:, act.size(1) // 2:])


class TabNetTransform(torch.nn.Module):
    def __init__(self, in_dim : int, out_dim : int, glu_act : bool = True):
        super().__init__()
        if glu_act: out_dim *= 2

        self.glu_act = glu_act

        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        h = self.bn(self.linear(x))
        if self.glu_act: h = glu(h)
        return h

class TabNet(torch.nn.Module):
    @dataclass
    class Config:
        cols : list[FeatureColumn]
        hidden_dim : int
        out_dim : int
        num_classes : int
        num_decision_steps : int
        relaxation_factor : float
        epsilon : float
        sparse_loss_weight : float

        @property
        def num_features(self): return len(self.cols)

    covertype_cols = [
        NumericColumn(0, 'Elevation'),
        NumericColumn(1, 'Aspect'),
        NumericColumn(2, 'Slope'),
        NumericColumn(3, 'Horizontal_Distance_To_Hydrology'),
        NumericColumn(4, 'Vertical_Distance_To_Hydrology'),
        NumericColumn(5, 'Horizontal_Distance_To_Roadways'),
        NumericColumn(6, 'Hillshade_9am'),
        NumericColumn(7, 'Hillshade_Noon'),
        NumericColumn(8, 'Hillshade_3pm'),
        NumericColumn(9, 'Horizontal_Distance_To_Fire_Points')
    ] + [
        BoolColumn(10 + i, f'Wilderness_Area{i + 1}')
        for i in range(4)
    ] + [
        BoolColumn(14 + i, f'Soil_Type{i + 1}')
        for i in range(40)
    ]

    covertype_config = Config(
        cols=covertype_cols,
        hidden_dim=4,
        out_dim=2,
        num_classes=7,
        num_decision_steps=6,
        relaxation_factor=1.5,
        epsilon=0.00001,
        sparse_loss_weight=0.0001
    )

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        self.encoders = torch.nn.ModuleList([
            col.make_encoder() for col in config.cols
        ])

        self.enc_bn = torch.nn.BatchNorm1d(config.num_features)

        self.tfrm1 = TabNetTransform(config.num_features, config.hidden_dim)
        self.tfrm2 = TabNetTransform(config.hidden_dim, config.hidden_dim)

        self.tfrm3 = torch.nn.ModuleList([
            TabNetTransform(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_decision_steps)
        ])

        self.tfrm4 = torch.nn.ModuleList([
            TabNetTransform(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_decision_steps)
        ])

        self.coef_trfm = torch.nn.ModuleList([
            TabNetTransform(config.out_dim, config.num_features, glu_act=False)
            for _ in range(config.num_decision_steps - 1)
        ])

        self.classifier = torch.nn.Sequential(*[
            torch.nn.Linear(config.out_dim, config.num_classes, bias=False),
            torch.nn.Softmax(dim=1)
        ])

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()


    def forward(self, cols : list[torch.Tensor]):
        x = torch.stack([enc(x).squeeze() for enc, x in zip(self.encoders, cols)], dim=1)
        x = self.enc_bn(x)

        batch_size = x.size(0)

        agg_out = []
        agg_mask = []
        agg_entropy = []

        masked_features = x
        mask_values = torch.zeros(
            [batch_size, self.config.num_features],
            dtype=x.dtype,
            device=x.device)

        compl_aggregated_mask_values = \
            torch.ones(
                (batch_size, self.config.num_features),
                dtype=x.dtype,
                device=x.device)

        for ni in range(self.config.num_decision_steps):

            h = self.tfrm1(masked_features)
            h = (self.tfrm2(h) + h) * np.sqrt(0.5)
            h = (self.tfrm3[ni](h) + h) * np.sqrt(0.5)
            h = (self.tfrm4[ni](h) + h) * np.sqrt(0.5)

            if ni > 0:
                decision_out = torch.relu(x[:, :self.config.out_dim])
                agg_out.append(decision_out)

                scale_agg = torch.sum(decision_out, dim=1, keepdim=True) \
                    / (self.config.num_decision_steps - 1)

                agg_mask.append(mask_values * scale_agg)

            features_for_coef = h[:, self.config.out_dim:]

            if ni < self.config.num_decision_steps - 1:
                mask_values = torch.softmax(
                    self.coef_trfm[ni](features_for_coef) * \
                        compl_aggregated_mask_values,
                    dim=1)

                compl_aggregated_mask_values = \
                    compl_aggregated_mask_values * \
                        (self.config.relaxation_factor - mask_values)

                agg_entropy.append(torch.mean(
                    torch.sum(
                        -mask_values * torch.log(mask_values + self.config.epsilon), dim=1)) \
                    / (self.config.num_decision_steps - 1))

                masked_features = mask_values * x

        out = torch.sum(torch.stack(agg_out, dim=0), dim=0)
        entropy = torch.sum(torch.stack(agg_entropy, dim=0), dim=0)

        return out, entropy

    def classify(self, cols : list[torch.Tensor]):
        encoded, _ = self.forward(cols)
        cls_out = self.classifier(encoded)
        return cls_out.argmax(dim=1) + 1


    def loss(self, cols : list[torch.Tensor], labels : torch.Tensor):
        encoded, entropy = self.forward(cols)
        cls_out = self.classifier(encoded)

        return self.cross_entropy_loss(cls_out, labels - 1) + \
            entropy * self.config.sparse_loss_weight
