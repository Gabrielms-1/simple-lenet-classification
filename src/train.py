# Importando as bibliotecas necessárias:
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn            # Módulo para construir redes neurais
import torch.optim as optim      # Módulo de otimizadores (ex: SGD, Adam)
from torch.utils.data import Dataset, DataLoader  # Para trabalhar com datasets customizados

from LeNet import LeNet, transformations

# Definindo o Dataset customizado:
class CustomImageDataset(Dataset):
    """
    Um dataset customizado que carrega imagens e suas respectivas classes a partir de um CSV.
    
    Parâmetros:
        csv_file (str): Caminho para o arquivo CSV com as anotações. 
                        Assume que a primeira coluna contém o nome do arquivo da imagem 
                        e a segunda coluna a classe correspondente.
        root_dir (str): Diretório onde as imagens estão armazenadas.
        transform (callable, opcional): Transformações a serem aplicadas nas imagens.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        # Lê o CSV com pandas:
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Retorna o tamanho do dataset
        return len(self.annotations)

    def __getitem__(self, idx):
        # Constrói o caminho completo para a imagem:
        img_path = self.annotations.iloc[idx, 0]
        # Abre a imagem com PIL e garante que ela tenha 3 canais (RGB):
        image = Image.open(img_path).convert('RGB')
        # Converte a anotação da classe para inteiro:
        label = int(self.annotations.iloc[idx, 1])
        
        # Aplica as transformações, se houver:
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

# Definindo as transformações que serão aplicadas nas imagens:


# Criando instância do dataset e DataLoader:
csv_file = 'data/skin_cancer.v2i.multiclass/train/processed_classes.csv'
root_dir = 'data/skin_cancer.v2i.multiclass/train'

dataset = CustomImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transformations)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

# Definindo a arquitetura LeNet:


# Instanciando o modelo:
num_classes = 8  # Atualize conforme o número de classes do seu dataset
model = LeNet(num_classes=num_classes)

# Configurando o dispositivo (GPU se disponível, senão CPU):
device = torch.device("cpu")
model.to(device)

# Definindo a função de perda e o otimizador:
criterion = nn.CrossEntropyLoss()  
# CrossEntropyLoss combina softmax com cálculo de entropia cruzada e é comum em problemas de classificação.

optimizer = optim.Adam(model.parameters(), lr=0.001)
# Otimizador Adam:
#   - model.parameters(): parâmetros do modelo a serem otimizados
#   - lr: taxa de aprendizado   

# Função para treinar o modelo:
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    """
    Função que executa o loop de treinamento.
    
    Parâmetros:
        model (nn.Module): Modelo a ser treinado.
        dataloader (DataLoader): DataLoader com os dados de treinamento.
        criterion: Função de perda.
        optimizer: Otimizador para atualizar os pesos.
        device: Dispositivo para computação (CPU ou GPU).
        num_epochs (int): Número de épocas de treinamento.
    """
    
    train_loss = []
    train_acc = []
    
    model.train()  # Coloca o modelo em modo de treinamento
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, path in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zera os gradientes para evitar acumulação
            
            outputs = model(images)  # Forward pass: computa as predições
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)  # Calcula a perda
            loss.backward()  # Backward pass: computa os gradientes
            optimizer.step()  # Atualiza os pesos
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Calcula a perda média da época:
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return train_loss, train_acc

def plot_metrics(train_losses, train_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot da Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.title('Curva de Loss durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot da Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r-o', label='Train Accuracy')
    plt.title('Curva de Acurácia durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    # Salva o plot em um arquivo
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    print("Plot das métricas salvo como training_metrics.png")  

# Iniciando o treinamento:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    
    args = parser.parse_args()
    
    num_epochs = args.epochs
    
    train_loss, train_acc = train_model(model, dataloader, criterion, optimizer, device, num_epochs)
    
    torch.save(model.state_dict(), 'data/lenet_model.pth')
    
    print("Model saved as lenet_model.pth")

    plot_metrics(train_loss, train_acc, num_epochs)
