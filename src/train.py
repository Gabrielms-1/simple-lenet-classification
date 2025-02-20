# Importing necessary libraries:
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import wandb

from LeNet import LeNet, transformations

class CustomImageDataset(Dataset):
    """
    A custom dataset that loads images and their respective classes from a CSV.
    
    Parameters:
        csv_file (str): Path to the CSV file with annotations. 
        root_dir (str): Directory where images are stored.
        transform (callable, optional): Transformations to be applied to the images.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        # Read the CSV with pandas:
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = int(self.annotations.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


train_csv_file = 'data/skin_cancer.v2i.multiclass/train/processed_classes.csv'
train_root_dir = 'data/skin_cancer.v2i.multiclass/train'

val_csv_file = 'data/skin_cancer.v2i.multiclass/valid/processed_classes.csv'
val_root_dir = 'data/skin_cancer.v2i.multiclass/valid'


train_dataset = CustomImageDataset(csv_file=train_csv_file, root_dir=train_root_dir, transform=transformations)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

val_dataset = CustomImageDataset(csv_file=val_csv_file, root_dir=val_root_dir, transform=transformations)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--batch_size', '-b', type=int, default=64)
args = parser.parse_args()

wandb.init(
project="skin-cancer-classification",
config={
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": 0.001
    }
)

num_classes = 8  
model = LeNet(num_classes=num_classes)

device = torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), wandb.config.learning_rate)

def evaluate_model(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)  
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)  
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
 
        return epoch_loss, epoch_acc


def train_model(model, dataloader, criterion, optimizer, device):
    """
    Function that executes the training loop.
    
    Parameters:
        model (nn.Module): Model to be trained.
        dataloader (DataLoader): DataLoader with training data.
        criterion: Loss function.
        optimizer: Optimizer to update weights.
        device: Device for computation (CPU or GPU).
    """
    
    train_loss = []
    train_acc = []
    
    model.train()  
       
    for epoch in range(wandb.config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, path in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  
            
            outputs = model(images)  
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{wandb.config.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc = evaluate_model(model, val_dataloader, device)

        wandb.log({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

    return train_loss, train_acc

def plot_metrics(train_losses, train_accuracies):
    epochs = range(1, wandb.config.epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.title('Loss Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r-o', label='Train Accuracy')
    plt.title('Accuracy Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    print("Metrics plot saved as training_metrics.png")  




if __name__ == '__main__':
    
    
    train_loss, train_acc = train_model(model, train_dataloader, criterion, optimizer, device)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f'data/lenet_model_{current_time}.pth')
    
    print(f"Model saved as lenet_model_{current_time}.pth")

    plot_metrics(train_loss, train_acc)
    wandb.finish()
