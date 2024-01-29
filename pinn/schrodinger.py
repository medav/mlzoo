import os
import torch
import numpy as np
from dataclasses import dataclass
from torch.autograd import grad

import scipy
from scipy.stats import qmc

class PinnSchrodinger(torch.nn.Module):
    @dataclass
    class Config:
        x_lb : torch.Tensor
        x_ub : torch.Tensor
        t_lb : torch.Tensor
        t_ub : torch.Tensor
        layers : list[int]

    def __init__(self, config : Config):
        super().__init__()
        self.register_buffer('x_lb', torch.tensor([config.x_lb]))
        self.register_buffer('x_ub', torch.tensor([config.x_ub]))
        self.register_buffer('t_lb', torch.tensor([config.t_lb]))
        self.register_buffer('t_ub', torch.tensor([config.t_ub]))
        self.register_buffer('lb', torch.tensor([config.x_lb, config.t_lb]))
        self.register_buffer('ub', torch.tensor([config.x_ub, config.t_ub]))

        self.layers = [2] + config.layers + [2]

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

    def forward_uv(self, x : torch.Tensor, t : torch.Tensor):
        x.requires_grad_(True)
        t.requires_grad_(True)

        h = 2.0 * (torch.cat([x, t], -1) - self.lb) / (self.ub - self.lb) - 1.0

        uv = self.nn(h)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]

        return u, v, u_x, v_x

    def forward_f(
        self,
        x : torch.Tensor,
        t : torch.Tensor,
        u : torch.Tensor,
        v : torch.Tensor,
        u_x : torch.Tensor,
        v_x : torch.Tensor
    ):
        u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        v_t = grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]

        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return f_u, f_v

    def forward(self, x : torch.Tensor, t : torch.Tensor):
        u, v, u_x, v_x = self.forward_uv(x, t)
        f_u, f_v = self.forward_f(x, t, u, v, u_x, v_x)
        return u, v, u_x, v_x, f_u, f_v

    def loss(
        self,
        x : torch.Tensor,
        t : torch.Tensor,
        u : torch.Tensor,
        v : torch.Tensor,
        xt_f : torch.Tensor
    ):
        u_pred, v_pred, _, _ = self.forward_uv(x, t)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.forward_uv(self.x_lb[None, :], self.t_lb[None, :])
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.forward_uv(self.x_ub[None, :], self.t_ub[None, :])
        _, _, _, _, f_u_pred, f_v_pred = self.forward(xt_f[:, 0:1], xt_f[:, 1:2])

        return \
            torch.mean(torch.square(u - u_pred)) + \
            torch.mean(torch.square(v - v_pred)) + \
            torch.mean(torch.square(u_lb_pred - u_ub_pred)) + \
            torch.mean(torch.square(v_lb_pred - v_ub_pred)) + \
            torch.mean(torch.square(u_x_lb_pred - u_x_ub_pred)) + \
            torch.mean(torch.square(v_x_lb_pred - v_x_ub_pred)) + \
            torch.mean(torch.square(f_u_pred)) + \
            torch.mean(torch.square(f_v_pred))

if __name__ == '__main__':
    N_train = 5000
    dtype = torch.float32
    dev = torch.device('cuda:0')

    noise = 0.0

    # Doman bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])


    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [100, 100, 100, 100]

    data = scipy.io.loadmat('./NLS.mat')

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    ###########################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)

    sampler = qmc.LatinHypercube(d=2)


    x = torch.tensor(x[idx_x, :]).to(dtype).to(dev)
    u = torch.tensor(Exact_u[idx_x, 0:1]).to(dtype).to(dev)
    v = torch.tensor(Exact_v[idx_x, 0:1]).to(dtype).to(dev)
    t = torch.tensor(t[idx_t, :]).to(dtype).to(dev)
    xt_f = torch.tensor(lb + (ub - lb) * sampler.random(n=N_f)).to(dtype).to(dev)

    h_dim = 100

    config = PinnSchrodinger.Config(
        x_lb = lb[0],
        x_ub = ub[0],
        t_lb = lb[1],
        t_ub = ub[1],
        layers=[h_dim] * 4
    )

    model = PinnSchrodinger(config).to(dtype).to(dev)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    for it in range(200000):
        optimizer.zero_grad()
        loss = model.loss(x, t, u, v, xt_f)
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print(f'It: {it}, Loss: {loss.item():.4f}')

    torch.save(model, './pinn-schr.pt')

