import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random


class LFW_dataset(torch.utils.data.Dataset):

  def __init__(self, images_dict, ids, split="train", transform = None):
    self.split = split
    self.transform = transform
    self.images_dict = images_dict
    self.ids = ids

  def __getitem__(self, index):
    id1 = self.ids[index]
    if len(self.images_dict[id1]) == 1:
      id2 = np.random.randint(0, len(self.ids))
      id2 = self.ids[id2]
      label = 0
    else:
      id2 = id1
      label = 1
      
    img1 = Image.open(self.images_dict[id1][0])
    img2 = Image.open(random.sample(self.images_dict[id2], 1)[0])

    if self.transform is not None:
      img1 = self.transform(img1)
      img2 = self.transform(img2)
    else:
      img1 = transforms.ToTensor()(img1)
      img2 = transforms.ToTensor()(img2)

    if label == 0:
      label = torch.Tensor([1, 0])
    else:
      label = torch.Tensor([0, 1])
    img = torch.cat((img1, img2), 0)
    return img, label

  def __len__(self):
    return len(self.ids)