import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ghostnet import ghostnet

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ghostnet(num_classes=37)
model.load_state_dict(torch.load('ghostnet_pet_model.pth'))
model = model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
               # (include all 37 class names here)
               'Yorkshire Terrier']

# Load and preprocess image
img_path = './Dataset/oxford-iiit-pet/images/Abyssinian_1.jpg'  # Change to your test image path
image = Image.open(img_path)
input_tensor = preprocess(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]

# Display result
plt.imshow(image)
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()
