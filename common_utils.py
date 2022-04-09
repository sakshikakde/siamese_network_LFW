import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
import time

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        
def calculate_accuracy(outputs, targets):

    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()

    _, labels = targets.topk(1, 1, True)
    labels = labels.t()

    correct = pred.eq(labels.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def adjust_learning_rate(optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr'] *= 0.5


def plot_loss(loss, title = "training loss"):
    plt.plot(loss, 'b', label = title)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

def plot_accuracy(accuracy, title = "training accuracy"):
    plt.plot(accuracy, 'b', label = title)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()