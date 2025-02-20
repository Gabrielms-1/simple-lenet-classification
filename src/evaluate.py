import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LeNet import LeNet, transformations
from train import CustomImageDataset
import argparse

def main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 8 
    model = LeNet(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    val_dataset = CustomImageDataset(csv_file='data/skin_cancer.v2i.multiclass/test/processed_classes.csv',
                                     transform=transformations,
                                     root_dir='data/skin_cancer.v2i.multiclass/test') 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, path in val_loader:
            image_name = path[0].split('/')[-1]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_dataset)
    val_accuracy = 100.0 * correct / total
 
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    args = parser.parse_args()
    main(args.model)
