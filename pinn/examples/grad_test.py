import torch
import numpy as np
from dataclasses import dataclass
from torch.autograd import grad

import tensorflow as tf


x = np.random.normal(size=(32, 128))
w = np.random.normal(size=(128, 128))

x_t = torch.tensor(x, requires_grad=True)
w_t = torch.tensor(w, requires_grad=True)

y_t = torch.matmul(x_t, w_t)

y_x_t = grad(y_t, x_t, torch.ones_like(y_t))[0]

x_tf = tf.convert_to_tensor(x)
w_tf = tf.convert_to_tensor(w)


@tf.function
def foo(x_tf, w_tf):
    y_tf = tf.matmul(x_tf, w_tf)

    y_x_tf = tf.gradients(y_tf, x_tf)[0]
    return y_x_tf

y_x_tf = foo(x_tf, w_tf)

print('Are they the same?', np.allclose(y_x_t.detach().numpy(), y_x_tf.numpy()))
