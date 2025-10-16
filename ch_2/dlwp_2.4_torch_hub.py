import torch
from torch import hub

resnet18_model = hub.load('pytorch/vision:main',
                          'resnet18',
                          pretrained=True)


