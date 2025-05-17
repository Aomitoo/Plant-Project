import torch
from models_utils_new import DiseaseClassifier
from torchvision import transforms
from PIL import Image


def predict(image_path):
    model = DiseaseClassifier(num_classes=68)
    model.load_state_dict(torch.load('models/doctorp_resnext_arcface.pth'))
    model.eval()

    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4467, 0.4889, 0.3267], std=[0.2299, 0.2224, 0.2289])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_index = predicted.item()

    
    class_name = class_names[class_index]
    return class_name