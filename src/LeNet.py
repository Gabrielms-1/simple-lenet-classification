import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    Implementation of the LeNet architecture for classification.
    
    Parameters:
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes=10, input_size=128):
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
        
        # Cálculo correto das dimensões após as operações convolucionais
        self.feature_size = ((input_size - 4) // 2 - 4) // 2  # (128-4=124→62 →62-4=58→29)
        self.fc1 = nn.Linear(16 * self.feature_size ** 2, 120)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(input=x, kernel_size=2, stride=2)

        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(input=x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x) 
        return x

