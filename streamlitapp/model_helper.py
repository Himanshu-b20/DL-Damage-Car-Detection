from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

trained_model = None
labels = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        #freeze all layers except the final fully connected layer
        for params in self.model.parameters():
            params.requires_grad= False

        #Unfreeze layer 4 and fcc
        for params in self.model.layer4.parameters():
            params.requires_grad= True

        ##Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
    def forward(self, x):
        x=self.model(x)
        return x

def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = image_transform(image).unsqueeze(0)  # (3, 224,224) but our model works on batches like (32, 3, 224,224) thats why we did unsquezze

    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load('model/saved_model.pth'))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _,predicted = torch.max(output, 1)

    return labels[predicted.item()]