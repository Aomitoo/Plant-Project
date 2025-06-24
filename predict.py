import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

class Config:
    NUM_CLASSES = 37
    IMG_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKBONE_FEATURE_DIM = 1536
    EMBEDDING_DIM = 1280
    DROPOUT = 0.35

class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, m=2.0):
        super().__init__()
        self.m = m
        self.scale = 10.0
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=1)
        cos_theta = torch.mm(x_norm, W_norm.t())
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cos_theta * self.scale
        acos_theta = torch.acos(cos_theta)
        m_acos = self.m * acos_theta
        floor_term = torch.floor(m_acos / math.pi).long()
        psi_theta = (-1) ** floor_term * torch.cos(m_acos) - 2 * floor_term
        psi_theta = torch.clamp(psi_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * psi_theta + (1.0 - one_hot) * cos_theta
        return output * self.scale

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, stage='stage2'):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(Config.BACKBONE_FEATURE_DIM, Config.EMBEDDING_DIM)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.stage = stage
        if stage == 'stage1':
            self.head = SphereFace(Config.EMBEDDING_DIM, num_classes, m=2.0)
        else:
            self.head = ClassificationHead(Config.EMBEDDING_DIM, num_classes)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.dropout(x)
        if self.stage == 'stage1':
            return self.head(x, labels)
        return self.head(x)

def predict_with_tta(image_path, num_augmentations=5):
    class_names = ['Alternaria leaf blight', 'Anthocyanosis', 'Anthracnose', 'Aphid', 'Aphid effects', 'Ascochyta blight', 'Bacterial spot', 'Black chaff', 'Black rot', 'Black spots', 'Blossom end rot', 'Botrytis cinerea', 'Burn', 'Downy mildew', 'Dry rot', 'Edema', 'Grey mold', 'Healthy', 'Late blight', 'Leaf deformation', 'Leaf miners', 'Loss of foliage turgor', 'Marginal leaf necrosis', 'Mealybug', 'Mechanical damage', 'Mosaic virus', 'Nutrient deficiency', 'Powdery mildew', 'Rust', 'Scale', 'Shot hole', 'Sooty mold', 'Spider mite', 'Thrips', 'Whitefly', 'Wilting', 'Yellow leaves']
    model = DiseaseClassifier()
    state_dict = torch.load('models/EfficientNet_B3_sphereface_60%_marg=1.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(Config.DEVICE)
    
    transform_base = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4425, 0.4931, 0.3288], std=[0.1961, 0.1912, 0.1884])
    ])
    transform_aug = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4425, 0.4931, 0.3288], std=[0.1961, 0.1912, 0.1884])
    ])
    
    image = Image.open(image_path).convert('RGB')
    images = [transform_base(image)]  # Original image
    for _ in range(num_augmentations - 1):
        images.append(transform_aug(image))  # Augmented versions
    images = torch.stack(images).to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        avg_probabilities = probabilities.mean(dim=0)
        
        top_probs, top_indices = torch.topk(avg_probabilities, 3)
        top_results = [
            {"class_name": class_names[idx.item()], "index": idx.item(), "probability": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return top_results

if __name__ == "__main__":
    result = predict_with_tta("2.jpg")
    for res in result:
        print(f"Class: {res['class_name']}, Index: {res['index']}, Probability: {res['probability']:.4f}")