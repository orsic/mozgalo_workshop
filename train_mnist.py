import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchvision.datasets import MNIST, CIFAR10
from sklearn.metrics import accuracy_score

from model import LogRegClassifier, ConvClassifier
from transforms import to_tensor, show_errors, show_fc_params, show_conv_params


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

def evaluate(dataloader, size, model):
    preds, true = np.zeros((size,), dtype=np.uint8), np.zeros((size,), dtype=np.uint8)
    for step, (x, y) in enumerate(dataloader):
        logits = model(x.to(device))
        _, pred = logits.max(1)
        pred = pred.byte().cpu().numpy()
        t = y.byte().cpu().numpy()
        preds[step * len(pred): (step + 1) * len(pred)] = pred
        true[step * len(pred): (step + 1) * len(pred)] = t
    return accuracy_score(true, preds), true, preds


# mnist_train = MNIST(os.getcwd(), train=True, transform=to_tensor, download=True)
# mnist_test = MNIST(os.getcwd(), train=False, transform=to_tensor, download=True)

cifar_train = CIFAR10(os.getcwd(), download=True, transform=to_tensor, train=True)
cifar_test = CIFAR10(os.getcwd(), download=True, transform=to_tensor, train=False)

# model instantiation
# model = LogRegClassifier(28 ** 2, 10).to(device)
model = ConvClassifier(in_channels=3, num_classes=10, input_size=(32, 32)).to(device)

# training loss and optimizer
# optimizer = optim.SGD(model.parameters(), lr=1e-1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# data batching
batch_size = 256
dataloader_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
dataloader_test = DataLoader(cifar_test, batch_size=batch_size)


class Trainer(object):
    def __init__(self, loader_train, loader_test, model, criterion, optimizer, epochs):
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.num_train = len(self.loader_train.dataset)
        self.num_test = len(self.loader_test.dataset)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def train(self):
        for e in range(self.epochs):
            steps = len(dataloader_train)
            for step, (x, y) in enumerate(self.loader_train):
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    print(f'Epoch {e+1}/{self.epochs} step {step}/{steps} loss: {loss.cpu().item():.2f}')
            acc_train, true_train, preds_train = evaluate(self.loader_train, self.num_train, self.model)
            acc_test, true_test, preds_test = evaluate(self.loader_test, self.num_test, self.model)
            print(f'Accuracy train: {100 * acc_train:.2f}%')
            print(f'Accuracy test: {100 * acc_test:.2f}%')
        return true_test, preds_test

# show_fc_params(model.W)
# train loop
with Trainer(dataloader_train, dataloader_test, model, criterion, optimizer, epochs=15) as trainer:
    true_test, preds_test = trainer.train()

# show_errors(mnist_test, true_test, preds_test)
# show_fc_params(model.W)
show_conv_params(model.model[0].weight)