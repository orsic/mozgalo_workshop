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
    return torch.from_numpy(np.expand_dims(x, axis=0))


def show_batch(batch, preds=None):
    imgs, labels = batch
    N, _, H, W = imgs.shape
    r, c = [int(math.ceil(math.sqrt(N)))] * 2
    if preds is None:
        preds = torch.ones(N).long() * -1

    for i, (im, label, pred) in enumerate(zip(imgs, labels, preds)):
        plt.subplot(r,c,i+1)
        plt.imshow(im.numpy().squeeze(), cmap='gray')
        plt.title(f'Label: {label.item()} Prediction: {pred.item()}')
    plt.tight_layout()
    plt.show()

def show_errors(dataset, true, pred):
    missedd_indices = np.where(pred != true)[0][:16]
    missed_img = torch.stack([dataset[i][0] for i in missedd_indices])
    missed_lab, missed_pred = true[missedd_indices], pred[missedd_indices]
    show_batch((missed_img, missed_lab), preds=missed_pred)