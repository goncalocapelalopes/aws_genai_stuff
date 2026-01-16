import torch.nn as nn
import torch.nn.functional as F

class TrivialCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # Your implementation here
        res = self.conv1(x)
        res = F.relu(res)
        res = self.conv2(res)
        res = F.relu(res)
        res = self.pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res