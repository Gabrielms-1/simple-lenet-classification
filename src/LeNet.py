import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class LeNet(nn.Module):
    """
    Implementation of the LeNet architecture for classification.
    
    Parameters:
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # First convolutional layer:
        #   - Input: 3 channels (RGB) 32x32
        #   - kernel_size: 5 (5x5 filter)
        #   - Output: 6 channels (number of filters) 28x28 (dimension reduced by kernel_size) with features extracted
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        
        # Second convolutional layer:
        #   - Input: 6 channels 14x14
        #   - Output: 16 channels 10x10 (dimension reduced by kernel_size) with features extracted
        #   - kernel_size: 5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Fully-connected layers:
        # Considering that the input has size 32x32, after two conv + pool operations,
        # the spatial dimension reduces to 5x5.
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(input=x, kernel_size=2, stride=2)

        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(input=x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = self.fc3(x) 
        return x

transformations = transforms.Compose([
    transforms.Resize((32, 32)),   # Resizes the image to 32x32 pixels (original size of LeNet)
    transforms.ToTensor(),         # Converts the PIL image to a PyTorch tensor with scale [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],   # Normalizes the color channels; 
                         std=[0.5, 0.5, 0.5])    # (mean and std can be adjusted according to the dataset)
])