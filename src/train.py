# Importing necessary libraries:
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import recall_score, f1_score
import os
from data_loader import FolderBasedDataset
from LeNet import LeNet


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
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

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
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        

        val_loss, val_acc, val_recall, val_f1 = evaluate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        wandb.log({
            "epoch": i+1,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_recall": val_recall,
            "val_f1": val_f1
        })
        # print metris in a table format    
        print("-"*100)
        print(f"Epoch {i+1}/{args.epochs}")
        print(f"Metrics: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print(f"Metrics: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f},  Recall: {val_recall:.4f},  F1: {val_f1:.4f}")
        print("-"*100)
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
                os.path.join(args.checkpoint_dir, f"checkpoint_{i+1}.pth")
            )
            print(f"Checkpoint {i+1} saved as {os.path.join(args.checkpoint_dir, f'checkpoint_{i+1}.pth')}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_best_f1.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "epoch": i+1,
                    "loss": epoch_loss,
                    "accuracy": epoch_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "optimizer_state_dict": optimizer.state_dict(), 
                    "wandb_config": wandb.config
                }, 
                best_checkpoint_path
            )
            print(f"Best F1 checkpoint saved as {best_checkpoint_path}")

    return train_losses, train_accuracies, val_losses, val_accuracies, val_recall, val_f1

def create_datasets(train_dir, val_dir, resize):
    train_dataset = FolderBasedDataset(train_dir, resize=resize)
    val_dataset = FolderBasedDataset(val_dir, resize=resize)
    
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False
    )
    return train_dataloader, val_dataloader

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, args):
    epochs = range(1, args.epochs + 1)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Loss Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r-o', label='Validation Accuracy')
    plt.title('Accuracy Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, "training_metrics.png"))
    plt.show()
    print(f"Metrics plot saved as {os.path.join(args.checkpoint_dir, 'training_metrics.png')}")  

def main(args):
    wandb.init(
        project="cad-lenet-alzheimer-brain-classification",
        mode=args.wandb_mode,
        name=f"lenet_model_{args.model_name}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_classes": args.num_classes,
            "resize": args.resize
        }
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset, val_dataset = create_datasets(args.train, args.val, args.resize)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, args.batch_size)

    model = LeNet(num_classes=args.num_classes, input_size=args.resize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=0.0001)

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
    }, f"{args.model_dir}/lenet_model.pth")

    print(f"Model saved as lenet_model.pth in model folder")

    print("Model directory contents (args.model_dir):", os.listdir(args.model_dir))
    print("Checkpoint directory contents (args.checkpoint_dir):", os.listdir(args.checkpoint_dir))
    print("Output folder contents:", os.listdir("/opt/ml/output/"))
    plot_metrics(train_loss, train_acc, val_loss, val_acc, args)
    wandb.finish()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="cad-lenet-alzheimer-brain-classification")
    parser.add_argument("--epochs", type=int, default=20)   
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--task", type=str, default="cad-simple-lenet-alzheimer-brain-classification")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])   
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])   
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/ml/checkpoints")

    args, _ = parser.parse_known_args()

    
    main(args)
