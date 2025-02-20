import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LeNet import LeNet, transformations
from train import CustomImageDataset


def main():
    # Defina o dispositivo (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicialize o modelo com o mesmo número de classes usado no treinamento
    num_classes = 8  # ajuste conforme seu dataset
    model = LeNet(num_classes=num_classes).to(device)

    # Carregue o estado salvo do modelo
    model.load_state_dict(torch.load('data/lenet_model.pth', map_location=device))
    model.eval()  # Coloca o modelo em modo de avaliação

    # Crie o DataLoader para o dataset de validação
    val_dataset = CustomImageDataset(csv_file='data/skin_cancer.v2i.multiclass/test/processed_classes.csv',
                                     transform=transformations,
                                     root_dir='data/skin_cancer.v2i.multiclass/test')  # ou outro conjunto de transformações
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Defina a função de perda para avaliação
    criterion = nn.CrossEntropyLoss()

    # Variáveis para acumular as métricas
    val_loss = 0.0
    correct = 0
    total = 0

    # Desativa o cálculo de gradiente para economizar memória
    with torch.no_grad():
        for images, labels, path in val_loader:
            image_name = path[0].split('/')[-1]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            
            # Obtém as previsões: índice com maior valor nos logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcula a perda média e a acurácia
    val_loss = val_loss / len(val_dataset)
    val_accuracy = 100.0 * correct / total
 
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

if __name__ == '__main__':
    main()
