import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class LeNet(nn.Module):
    def __init__(self, num_classes=8):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class InferenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file, header=None)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        path = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/skin_cancer.v2i.multiclass/test/processed_classes.csv')
    parser.add_argument('--model', type=str, default='data/lenet_model_2.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = InferenceDataset(csv_file=args.csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    model = LeNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            outputs = model(images.to(args.device))
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            for i in range(len(paths)):
                confidence = probs[i][preds[i]].item()
                print(f"{paths[i]} - {preds[i].item()} - {labels[i].item()} - {confidence:.4f}")
