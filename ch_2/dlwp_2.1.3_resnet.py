import torch
from torchvision import models
from torchsummary import summary
from pprint import pprint

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet101(pretrained=True).to(device)
pprint(resnet)

resnet.eval()
summary(resnet, input_size=(3, 448, 448))

