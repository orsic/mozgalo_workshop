import torch
import numpy as np
from PIL import Image as pimg
import math
import matplotlib.pyplot as plt


def random_flip(image: pimg):
    if np.random.randint(0, 2) == 1:
        return image.transpose(pimg.FLIP_LEFT_RIGHT)
    return image


def to_tensor(image):
    x = np.float32(image)
    x /= 255
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=0)
    else:
        x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def show_batch(batch, preds=None, class_def=None):
    imgs, labels = batch
    N, C, H, W = imgs.shape
    r, c = [int(math.ceil(math.sqrt(N)))] * 2
    if preds is None:
        preds = torch.ones(N).long() * -1

    for i, (im, label, pred) in enumerate(zip(imgs, labels, preds)):
        plt.subplot(r, c, i + 1)
        plt.imshow(im.permute((1, 2, 0)).numpy().squeeze(), cmap='gray' if C == 1 else None)
        title = f'Label: {label.item() if class_def is None else class_def[int(label.item())]}'
        if pred > 0:
            title +=  f' Prediction: {pred.item() if class_def is None else class_def[int(pred.item())]}'
        plt.title(title)
    plt.tight_layout()
    plt.show()


def show_errors(dataset, true, pred, class_def=None):
    missedd_indices = np.where(pred != true)[0][:16]
    missed_img = torch.stack([dataset[i][0] for i in missedd_indices])
    missed_lab, missed_pred = true[missedd_indices], pred[missedd_indices]
    show_batch((missed_img, missed_lab), preds=missed_pred, class_def=class_def)


def show_fc_params(params, size=(28, 28)):
    W = np.copy(params.data.cpu().numpy().transpose())
    r, c = [int(math.ceil(math.sqrt(W.shape[0])))] * 2
    for i, w in enumerate(W):
        w = w.reshape(size) - w.min()
        w /= w.max()
        plt.subplot(r, c, i + 1)
        plt.imshow(w, cmap='gray')
    plt.show()


def show_conv_params(params):
    W = np.copy(params.data.cpu().numpy())
    r, c = [int(math.ceil(math.sqrt(W.shape[0])))] * 2
    for i, w in enumerate(W):
        w -= w.min()
        w /= w.max()
        plt.subplot(r, c, i + 1)
        plt.imshow(w.transpose(1, 2, 0).squeeze())
    plt.show()
