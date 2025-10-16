import torch
from PIL import Image
from torchvision import models
from torchvision.models import alexnet, AlexNet_Weights
from torchsummary import summary
from pprint import pprint

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load pretrained weights
weights = AlexNet_Weights.IMAGENET1K_V1
model = alexnet(weights=weights).to(device).eval()
pprint(model)

preprocess = weights.transforms()

# 2. Load and preprocess image
image_paths = [
    "../test_images/bobby.jpg",
    "../test_images/horse.jpg",
    "../test_images/image_1.png",
    "../test_images/image_2.png",
    "../test_images/image_3.png",
    "../test_images/image_4.png",
    "../test_images/image_5.png",
    "../test_images/zebra.jpg",
]

for path in image_paths:
    print(f"{path}")
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    
# 3. Run inference
    with torch.inference_mode():
        logits = model(x)
    probs = logits.softmax(dim=1)

# 4. Show top 5 predictions
    categories = weights.meta["categories"]
    top5 = torch.topk(probs, 5).indices[0].tolist()
    for rank, idx in enumerate(top5, 1):
        print(f"{rank}, {categories[idx]} - {probs[0, idx].item():.3f}")
    print("==========")


