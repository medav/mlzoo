import os
import torch
import numpy as np
from dataclasses import dataclass
from torch.autograd import grad

import scipy

class PinnNs(torch.nn.Module):
    @dataclass
    class Config:
        lb : torch.Tensor
        ub : torch.Tensor
        layers : list[int]

    def __init__(self, config : Config):
        super().__init__()
        self.register_buffer('lb', config.lb)
        self.register_buffer('ub', config.ub)
        self.layers = [3] + config.layers + [2]

        self.nn = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Linear(self.layers[i], self.layers[i + 1]),
                torch.nn.Tanh()
            ) for i in range(len(self.layers) - 2)],
            torch.nn.Linear(self.layers[-2], self.layers[-1])
        )

        for name, param in self.nn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor):
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        h = 2.0 * (torch.cat([x, y, t], 1) - self.lb) / (self.ub - self.lb) - 1.0

        psi_and_p = self.nn(h)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x, torch.ones_like(psi), create_graph=True)[0]

        u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, torch.ones_like(u), create_graph=True)[0]

        u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

        v_t = grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, torch.ones_like(v), create_graph=True)[0]

        v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

        p_x = grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, torch.ones_like(p), create_graph=True)[0]

        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss(
        self,
        x : torch.Tensor,
        y : torch.Tensor,
        t : torch.Tensor,
        u : torch.Tensor,
        v : torch.Tensor
    ):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.forward(x, y, t)

        return \
            torch.sum(torch.square(u - u_pred)) + \
            torch.sum(torch.square(v - v_pred)) + \
            torch.sum(torch.square(f_u_pred)) + \
            torch.sum(torch.square(f_v_pred))

if __name__ == '__main__':
    N_train = 5000
    dtype = torch.float32
    dev = torch.device('cuda:0')

    data = scipy.io.loadmat('./cylinder_nektar_wake.mat')

    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T

    UU = U_star[:, 0, :] # N x T
    VV = U_star[:, 1, :] # N x T
    PP = P_star # N x T

    x = XX.flatten()[:, None] # NT x 1
    y = YY.flatten()[:, None] # NT x 1
    t = TT.flatten()[:, None] # NT x 1

    u = UU.flatten()[:, None] # NT x 1
    v = VV.flatten()[:, None] # NT x 1
    p = PP.flatten()[:, None] # NT x 1

    np.random.seed(1234)
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = torch.tensor(x[idx, :], dtype=dtype).to(dev)
    y_train = torch.tensor(y[idx, :], dtype=dtype).to(dev)
    t_train = torch.tensor(t[idx, :], dtype=dtype).to(dev)
    u_train = torch.tensor(u[idx, :], dtype=dtype).to(dev)
    v_train = torch.tensor(v[idx, :], dtype=dtype).to(dev)

    X = torch.cat([x_train, y_train, t_train], 1)

    h_dim = 20

    config = PinnNs.Config(
        lb=torch.min(X, 0).values,
        ub=torch.max(X, 0).values,
        layers=[h_dim] * 8
    )

    model = PinnNs(config).to(dtype).to(dev)
    model = torch.compile(model)
    torch.save(model.state_dict(), './pinn-ns.pt')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    for it in range(200000):
        optimizer.zero_grad()
        loss = model.loss(x_train, y_train, t_train, u_train, v_train)
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print(f'It: {it}, Loss: {loss.item():.4f}, l1: {model.lambda_1.item()}, l2: {model.lambda_2.item()}')


    torch.save(model.state_dict(), './pinn-ns.pt')


