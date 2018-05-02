import torch
import torch.nn as nn


class LogRegClassifier(nn.Module):
    def __init__(self, num_features, num_classes, bias=False):
        super(LogRegClassifier, self).__init__()
        self.W = nn.Parameter(data=torch.randn((num_features, num_classes)))
        if bias:
            self.b = nn.Parameter(data=torch.zeros(num_classes))

    def forward(self, input):
        x = input.view(input.size(0), -1)
        s = torch.mm(x, self.W)
        if hasattr(self, 'b'):
            s += self.b
        return s


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=(28, 28)):
        super(ConvClassifier, self).__init__()
        fc_in_h, fc_in_w = self._flat_wh(input_size)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5),  # 24x24
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2),  # 11x11
            nn.Conv2d(16, 32, 5),  # 7x7
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2),  # 3x3
            Flatten(),  # 3*3*32
            nn.Linear(fc_in_h * fc_in_w * 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def _flat_wh(self, size):
        w, h = size
        return self._fc_in(h), self._fc_in(w)

    def _fc_in(self, x):
        c1 = x - 4
        p1 = (c1 - 3) // 2 + 1
        c2 = p1 - 4
        p2 = (c2 - 3) // 2 + 1
        return p2

    def forward(self, x):
        return self.model(x)
