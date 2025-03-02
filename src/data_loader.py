import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FolderBasedDataset(Dataset):
    """
    A custom dataset that loads images and their respective classes from a folder.
    
    Parameters:
        root_dir (str): Directory where images are stored.
        transform (callable, optional): Transformations to be applied to the images.
    """
    def __init__(self, root_dir, resize):
        self.images = self.get_image_path(root_dir)
        self.annotations = self.get_labels(root_dir)
        self.root_dir = root_dir
        self.resize = resize
        self.transform = self.get_transformations()
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

    def get_transformations(self):
        transformations = transforms.Compose([
            transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.LANCZOS),   # Resizes the image to 32x32 pixels (original size of LeNet)
            transforms.ToTensor(),  
            # transforms.Grayscale(num_output_channels=1),       # Converts the PIL image to a PyTorch tensor with scale [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],   # Normalizes the color channels; 
                             std=[0.5, 0.5, 0.5])    # (mean and std can be adjusted according to the dataset)
            ])
        return transformations
