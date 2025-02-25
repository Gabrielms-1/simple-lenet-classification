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
#import wandb
from sklearn.metrics import recall_score, f1_score
import os
import glob

from LeNet import LeNet, transformations

class CustomImageDataset(Dataset):
    """
    A custom dataset that loads images and their respective classes from a CSV.
    
    Parameters:
        csv_file (str): Path to the CSV file with annotations. 
        root_dir (str): Directory where images are stored.
        transform (callable, optional): Transformations to be applied to the images.
    """
    def __init__(self, root_dir, transform=None):
        self.images = self.get_image_path(root_dir)
        self.annotations = self.get_labels(root_dir)
        self.root_dir = root_dir
        self.transform = transform

        self.label_map_to_int = {label: i for i, label in enumerate(sorted(set(label for label in self.annotations)))}
        self.int_to_label_map = {i: label for label, i in self.label_map_to_int.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = (img_path).split("/")[-2]
        
        label = self.label_map_to_int[label]

        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

    def get_labels(self, root_dir):
        all_items = glob.glob(os.path.join(root_dir, '**', '*', '*.jpg'), recursive=True)
        labels = [item.split("/")[-2] for item in all_items]
        
        return labels

    def get_image_path(self, root_dir):
        all_items = glob.glob(os.path.join(root_dir, '**', '*', '*.jpg'), recursive=True)
        
        return all_items

def create_datasets(train_dir, val_dir, transform):
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)
    
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_dataloader, val_dataloader

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
 
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        return epoch_loss, epoch_acc, recall, f1

def train_model(model, dataloader, criterion, optimizer, device, val_dataloader):
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
    best_f1 = 0
       
    for i in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in dataloader:
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
        
        print(f"Epoch {i+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc, val_recall, val_f1 = evaluate_model(model, val_dataloader, criterion, device)


        # wandb.log({
        #     "epoch": i+1,
        #     "loss": epoch_loss,
        #     "accuracy": epoch_acc,
        #     "val_loss": val_loss,
        #     "val_accuracy": val_acc,
        #     "val_recall": val_recall,
        #     "val_f1": val_f1
        # })

        print(f"Validation Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")

        # checkpoint_path = f"{args.checkpoint_dir}lenet_model_{args.current_time}_checkpoint_{i+1}.pth"

        if (i + 1) % 10 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "epoch": i+1,
                    "loss": epoch_loss,
                    "accuracy": epoch_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, 
                f"/opt/ml/model/lenet_model_{args.current_time}_checkpoint_{i+1}.pth"
            )
            print(f"Checkpoint {i+1} saved as /opt/ml/model/lenet_model_{args.current_time}_checkpoint_{i+1}.pth")

        # if val_f1 > best_f1:
        #     best_f1 = val_f1
        #     checkpoint_path = f"{args.checkpoint_dir}lenet_model_{args.current_time}_checkpoint_best_f1.pth"
        #     torch.save(
        #         {
        #             "model_state_dict": model.state_dict(), 
        #             "epoch": i+1,
        #             "loss": epoch_loss,
        #             "accuracy": epoch_acc,
        #             "val_loss": val_loss,
        #             "val_accuracy": val_acc,
        #             "optimizer_state_dict": optimizer.state_dict(), 
        #             "wandb_config": wandb.config
        #         }, 
        #         checkpoint_path
        #     )
        #     print(f"Best F1 checkpoint saved as {checkpoint_path}")


    return train_loss, train_acc, val_loss, val_acc, val_recall, val_f1

def plot_metrics(train_losses, train_accuracies):
    epochs = range(1, args.epochs + 1)
    
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
    plt.savefig("/opt/ml/output/training_metrics.png")
    plt.show()
    print("Metrics plot saved as training_metrics.png")  

def main(args):
    # wandb.init(
    #     project=configs["wandb_project"],
    #     name=f"lenet_model_{configs['current_time']}",
    #     config={
    #         "epochs": args.epochs,
    #         "batch_size": args.batch_size,
    #         "learning_rate": configs["learning_rate"]
    #     }
    # )

    train_dataset, val_dataset = create_datasets(args.train, args.val, transformations)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, args.batch_size)

    model = LeNet(num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    if not os.path.exists("/opt/ml/model"):
        os.makedirs("/opt/ml/model")

    train_loss, train_acc, val_loss, val_acc, val_recall, val_f1 = train_model(model, train_dataloader, criterion, optimizer, device, val_dataloader)
    torch.save(
    {
        "model_state_dict": model.state_dict(), 
        "epoch": args.epochs,
        "loss": train_loss[-1],
        "accuracy": train_acc[-1],
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "optimizer_state_dict": optimizer.state_dict(),
        "int_to_label_map": train_dataset.int_to_label_map,  # Add this line
        "num_classes": args.num_classes,  # Add this line
    }, f"/opt/ml/checkpoints/lenet_model_1.pth")

    print(f"Model saved as lenet_model_1.pth")

    plot_metrics(train_loss, train_acc)
    # wandb.finish()

if __name__ == '__main__':
    
   
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="cad-unet-camvid-segmentation")
    parser.add_argument("--epochs", type=int, default=20)   
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--task", type=str, default="cad-simple-lenet-fashionmist")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])   
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])   
    parser.add_argument("--num_classes", type=int, default=10)
    args, _ = parser.parse_known_args()
    args.current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    main(args)
