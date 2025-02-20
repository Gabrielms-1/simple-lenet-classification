import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
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

transformations = transforms.Compose([
    transforms.Resize((32, 32)),   # Redimensiona a imagem para 32x32 pixels (tamanho original do LeNet)
    transforms.ToTensor(),         # Converte a imagem PIL para um tensor do PyTorch com escala [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],   # Normaliza os canais de cor; 
                         std=[0.5, 0.5, 0.5])    # (mean e std podem ser ajustados conforme o dataset)
])