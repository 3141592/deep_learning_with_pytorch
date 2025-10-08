import torch
from torchvision import models
from torchvision import transforms
from torchsummary import summary
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet101(pretrained=True).to(device)
pprint(resnet)

resnet.eval()
summary(resnet, input_size=(3, 448, 448))

print("Preprocess Images")
preprocess = transforms.Compose([
  transforms.Lambda(lambda im: im.convert('RGB')),  # force 3 channels
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )])

img = Image.open("../test_images/image_1.png").convert('RGB')
img
img.show()
img_t = preprocess(img)

# Undo normalization for display
unnormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)
img_disp = unnormalize(img_t).clamp(0, 1)  # keep pixel values in [0,1]

plt.imshow(img_disp.permute(1, 2, 0))  # convert CHW → HWC
plt.axis('off')
plt.show()

batch_t = torch.unsqueeze(img_t, 0)



