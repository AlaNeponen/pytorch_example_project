import torch.nn as nn
from torch.nn import Module

# Based on LeNet
class Cnn(Module):
    def __init__(self, numChannels, classes):
        super(Cnn, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=800, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=classes),
            nn.LogSoftmax(dim=1)
        
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x