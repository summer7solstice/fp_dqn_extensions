import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class BasicDQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(BasicDQNModel, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.convolution(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.convolution(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class DuelingDQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQNModel, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # divided former path to two paths, one for value, one for advantages
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.convolution(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.convolution(fx).view(fx.size()[0], -1)
        return self.fc_advantage(conv_out), self.fc_value(conv_out)

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))
