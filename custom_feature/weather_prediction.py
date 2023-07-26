# temperature prediction model with resnet101 using ordinal regression

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from scipy.stats import norm
import numpy as np

num_epochs = 30
learning_rate = 0.001
log_interval = 1

df = pd.read_csv('./data/final_data.csv')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, data_frame, transform=None, istrain=True):
        self.data_frame = data_frame
        self.transform = transform
        self.istrain = istrain

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_id = self.data_frame.iloc[idx, 0]
        image_path = "./img/clothes/" + str(image_id) + ".jpg"  # Adjust the path to your image directory
        image = Image.open(image_path)  # Implement a function to load images
        label = self.data_frame.iloc[idx, 1] - 1  # Adjust labels to start from 0 for ordinal regression
        y = np.zeros(12)
        y[label] = 1

        if self.transform:
            image = self.transform(image)
        if self.istrain:
            return image, y
        else:
            return image, label

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = CustomDataset(train_df, transform=transform_train)
val_dataset = CustomDataset(val_df, transform=transform_val, istrain=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = torchvision.models.resnet50(pretrained=True)

num_classes = 12
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % log_interval == log_interval - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / log_interval:.4f}")
            running_loss = 0.0
            print(f"--- {(time.time() - start_time):.4f} seconds ---")
            start_time = time.time()

    # Validation after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), './checkpoint/temperature_prediction.pth')
    print(f"Validation Accuracy: {accuracy:.2f}%")



