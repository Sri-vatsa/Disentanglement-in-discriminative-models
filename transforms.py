import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from utils import *

def _simclr_transform(size, s=1):
  color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
  data_transforms = T.Compose([T.RandomResizedCrop(size=size),
                                              T.RandomHorizontalFlip(),
                                              T.RandomApply([color_jitter], p=0.8),
                                              T.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              T.ToTensor()])
  return data_transforms

def _default_transform(img_size):
   data_transforms = T.Compose( [T.Resize(img_size),
                                 T.ToTensor(),
                                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
   return data_transforms


def get_transforms(is_contrastive, img_size, n_views):
  if is_contrastive:
    transform = ContrastiveLearningViewGenerator(_simclr_transform(img_size), n_views)
    return transform
  else:
    return _default_transform(img_size)