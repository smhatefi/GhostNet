import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
from model.ghostnet import ghostnet

# Hyperparameters and configurations
batch_size = 32
learning_rate = 0.001
num_epochs = 50
patience = 5  # For early stopping

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Oxford-IIIT Pet Dataset
data_dir = './Dataset'
train_dataset = datasets.OxfordIIITPet(root=data_dir, split='trainval', target_types='category', download=True, transform=transform)
test_dataset = datasets.OxfordIIITPet(root=data_dir, split='test', target_types='category', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ghostnet(num_classes=37).to(device)  # 37 classes in Oxford-IIIT Pet Dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Early stopping and training process
best_acc = 0.0
early_stop_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, patience):
    global best_acc, early_stop_counter, best_model_wts
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-" * 10}')
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss /= len(test_loader.dataset)
        val_acc = val_corrects.double() / len(test_loader.dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            torch.save(best_model_wts, 'ghostnet_pet_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, patience)
