# icg/data/transforms.py
from torchvision import transforms

# Hardcoded image sizes (ImageNet-style)
IMG_SIZE = 256
CROP_SIZE = 224

# Imagenet mean/std for EfficientNet encoder
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
