# Importando as bibliotecas necessárias:
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn            # Módulo para construir redes neurais
import torch.optim as optim      # Módulo de otimizadores (ex: SGD, Adam)
from torch.utils.data import Dataset, DataLoader  # Para trabalhar com datasets customizados
from torchvision import transforms  # Para realizar transformações (pré-processamento) nas imagens

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
        
        return image, label

# Definindo as transformações que serão aplicadas nas imagens:
transformations = transforms.Compose([
    transforms.Resize((32, 32)),   # Redimensiona a imagem para 32x32 pixels (tamanho original do LeNet)
    transforms.ToTensor(),         # Converte a imagem PIL para um tensor do PyTorch com escala [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],   # Normaliza os canais de cor; 
                         std=[0.5, 0.5, 0.5])    # (mean e std podem ser ajustados conforme o dataset)
])

# Criando instância do dataset e DataLoader:
csv_file = 'data/skin_cancer.v2i.multiclass/train/processed_classes.csv'
root_dir = 'data/skin_cancer.v2i.multiclass/train'

dataset = CustomImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transformations)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Definindo a arquitetura LeNet:
class LeNet(nn.Module):
    """
    Implementação da arquitetura LeNet para classificação.
    
    Parâmetros:
        num_classes (int): Número de classes de saída.
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # Primeira camada convolucional:
        #   - Entrada: 3 canais (RGB) 32x32
        #   - kernel_size: 5 (filtro 5x5)
        #   - Saída: 6 canais (quantidade de filtros) 28x28 (dimensão reduzida pelo kernel_size) com as características extraídas
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        
        # Camada de pooling (reduz a dimensionalidade):
        #   - kernel_size: 2 (janela 2x2)
        #   - stride: 2 (passo da janela)
        #   - Entrada: 6 canais 28x28
        #   - Saída: 6 canais 14x14 (dimensão reduzida pelo kernel_size e stride)
        
        
        # Segunda camada convolucional:
        #   - Entrada: 6 canais 14x14
        #   - Saída: 16 canais 10x10 (dimensão reduzida pelo kernel_size) com as características extraídas
        #   - kernel_size: 5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Camadas totalmente conectadas (fully-connected):
        # Considerando que a entrada tem tamanho 32x32, após duas operações de conv + pool,
        # a dimensão espacial reduz para 5x5.
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # Aplica a primeira convolução, ReLU e pooling:
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(input=x, kernel_size=2, stride=2)
        # Aplica a segunda convolução, ReLU e pooling:
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(input=x, kernel_size=2, stride=2)
        # "Flatten": transforma os mapas de características 2D em um vetor 1D
        x = x.view(x.size(0), -1)
        # Primeira camada totalmente conectada com ReLU:
        x = torch.relu(self.fc1(x))
        # Segunda camada totalmente conectada com ReLU:
        x = torch.relu(self.fc2(x))
        # Camada de saída (não é aplicada ativação pois vamos usar CrossEntropyLoss que aplica softmax internamente)
        x = self.fc3(x)
        return x

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
    model.train()  # Coloca o modelo em modo de treinamento
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Iniciando o treinamento:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=10)
    args = parser.parse_args()
    num_epochs = args.epochs
    train_model(model, dataloader, criterion, optimizer, device, num_epochs)
    torch.save(model.state_dict(), 'data/lenet_model.pth')
    print("Model saved as lenet_model.pth")
