import torch
from PIL import Image
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import os

# Definir a mesma arquitetura LeNet usada no treinamento
class LeNet(nn.Module):
    def __init__(self, num_classes=8):  # Garantir o mesmo número de classes
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Carregar o modelo treinado
def load_model(model_path, num_classes=8):
    model = LeNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Transformações (deve ser igual ao pré-processamento do treinamento)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Função para fazer previsões
def predict(image_path, model, class_names=None):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Adicionar dimensão do batch
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    if class_names:
        return predicted_class, class_names[predicted_class], confidence
    return predicted_class, confidence

# Carregar mapeamento de classes (se disponível)
def load_class_names(csv_path):
    df = pd.read_csv(csv_path)
    return df.iloc[:, 1].tolist()  # Assume que a segunda coluna tem os nomes

if __name__ == '__main__':
    # Configurações
    MODEL_PATH = 'data/lenet_model.pth'
    CLASS_CSV = 'data/skin_cancer.v2i.multiclass/test/_classes.csv'
    TEST_DIR = 'data/skin_cancer.v2i.multiclass/test/'
    
    # Carregar modelo e classes
    model = load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_CSV) if CLASS_CSV else None
    
    # Processar todas as imagens do diretório de teste
    correct = 0
    total = 0
    
    for img_file in os.listdir(TEST_DIR):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(TEST_DIR, img_file)
            
            # Fazer previsão
            class_idx, class_name, confidence = predict(img_path, model, class_names)
            
            # Exibir resultados
            print(f'Image: {img_file}')
            print(f'Predicted: {class_idx} ({class_name}) Confidence: {confidence:.2%}')
            print('-' * 50)
            
            total += 1

    print(f'\nTotal images processed: {total}')
