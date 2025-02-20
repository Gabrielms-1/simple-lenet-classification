# Importando as bibliotecas necessárias:
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb

from LeNet import LeNet, transformations

class CustomImageDataset(Dataset):
    """
    Um dataset customizado que carrega imagens e suas respectivas classes a partir de um CSV.
    
    Parâmetros:
        csv_file (str): Caminho para o arquivo CSV com as anotações. 
        root_dir (str): Diretório onde as imagens estão armazenadas.
        transform (callable, opcional): Transformações a serem aplicadas nas imagens.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        # Lê o CSV com pandas:
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


csv_file = 'data/skin_cancer.v2i.multiclass/train/processed_classes.csv'
root_dir = 'data/skin_cancer.v2i.multiclass/train'

dataset = CustomImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transformations)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

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

def train_model(model, dataloader, criterion, optimizer, device):
    """
    Função que executa o loop de treinamento.
    
    Parâmetros:
        model (nn.Module): Modelo a ser treinado.
        dataloader (DataLoader): DataLoader com os dados de treinamento.
        criterion: Função de perda.
        optimizer: Otimizador para atualizar os pesos.
        device: Dispositivo para computação (CPU ou GPU).
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

        wandb.log({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "accuracy": epoch_acc
        })

    return train_loss, train_acc

def plot_metrics(train_losses, train_accuracies):
    epochs = range(1, wandb.config.epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.title('Curva de Loss durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r-o', label='Train Accuracy')
    plt.title('Curva de Acurácia durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    print("Plot das métricas salvo como training_metrics.png")  




if __name__ == '__main__':
    
    
    train_loss, train_acc = train_model(model, dataloader, criterion, optimizer, device)
    
    torch.save(model.state_dict(), 'data/lenet_model.pth')
    
    print("Model saved as lenet_model.pth")

    plot_metrics(train_loss, train_acc)
    wandb.finish()
