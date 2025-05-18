import torch
from models_utils_new import DiseaseClassifier
from torchvision import transforms
from PIL import Image


def predict(image_path):
    class_names = ['Alternaria leaf blight', 'Anthocyanosis', 'Anthracnose', 'Ants', 'Aphid', 'Aphid effects', 'Ascochyta blight', 'Bacterial spot', 'Black chaff', 'Black rot', 'Black spots', 'Blossom end rot', 'Botrytis cinerea', 'Burn', 'Canker', 'Caterpillars', 'Cherry leaf spot', 'Coccomyces of pome fruits', 'Colorado beetle', 'Colorado beetle effects', 'Corn downy mildew', 'Cyclamen mite', 'Downy mildew', 'Dry rot', 'Edema', 
'Esca', 'Eyespot', 'Frost cracks', 'Galls', 'Grey mold', 'Gryllotalpa', 'Gryllotalpa effects', 'Healthy', 'Late blight', 'Leaf deformation', 'Leaf miners', 'Leaf spot', 'Leaves scorch', 'Lichen', 'Loss of foliage turgor', 'Marginal leaf necrosis', 'Mealybug', 'Mechanical damage', 'Monilia', 'Mosaic virus', 'Northern leaf blight', 'Nutrient deficiency', 'Pear blister mite', 'Pest damage', 'Polypore', 'Powdery mildew', 'Rust', 'Scab', 'Scale', 'Shot hole', 'Shute', 'Slugs', 'Slugs caterpillars effects', 'Sooty mold', 'Spider mite', 'Thrips', 'Tubercular necrosis', 'Verticillium wilt', 'Whitefly', 'Wilting', 'Wireworm', 'Wireworm effects', 'Yellow leaves']

    model = DiseaseClassifier(num_classes=68)
    model.load_state_dict(torch.load('models/Resnext_arcface_99.88%.pth'))
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