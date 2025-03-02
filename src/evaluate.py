import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from LeNet import LeNet, transformations
from train import CustomImageDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Evaluate LeNet model on image directory')
    parser.add_argument('--image_dir', type=str, default="data/fashionmnist/small/test")
    parser.add_argument('--model_path', type=str, default="data/checkpoints/lenet_model_2025-02-25_23-25-16_checkpoint_100.pth")
    parser.add_argument('--save_dir', type=str, default="data/results")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Recreate model architecture
    model = LeNet(num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = CustomImageDataset(root_dir=args.image_dir, transform=transformations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Lists to store all labels and predictions
    y_true = []
    y_pred = []
    
    # Run predictions
    results = []
    with torch.no_grad():
        for images, labels, img_paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Store batch results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            for path, pred in zip(img_paths, preds.cpu().numpy()):
                results.append({
                    "image_path": path,
                    "predicted_class": dataset.int_to_label_map[pred],
                    "predicted_class_idx": pred
                })


    # Save results to CSV
    df = pd.DataFrame(results)
    save_path = os.path.join(args.save_dir, "predictions.csv")
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    # Calculate and print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=dataset.int_to_label_map.values()))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()
