import torch
import torch.nn as nn


class LogRegClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogRegClassifier, self).__init__()
        self.W = nn.Parameter(data=torch.randn((num_features, num_classes)))
        self.b = nn.Parameter(data=torch.zeros(num_classes))

    def forward(self, input):
        x = input.view(input.size(0), -1)
        s = torch.mm(x, self.W) + self.b
        return s


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # 24x24
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2),  # 11x11
            nn.Conv2d(16, 32, 5),  # 7x7
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2),  # 3x3
            Flatten(),  # 3*3*32
            nn.Linear(3 * 3 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.model(x)
