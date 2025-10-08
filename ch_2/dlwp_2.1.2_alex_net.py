import torch
from torchvision import models
from torchsummary import summary
from pprint import pprint

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alexnet = models.AlexNet().to(device)
pprint(alexnet)

alexnet.eval()
summary(alexnet, input_size=(3, 448, 448))

