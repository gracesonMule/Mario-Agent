import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MarioCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        input_shape: tuple of (Channels, Height, Width)
        """
        super(MarioCNN, self).__init__()
        
        # --- Feature Extraction ---
        # Grouping layers into nn.Sequential makes the forward pass cleaner
        # and allows us to easily measure the output shape.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # --- Dynamic Dimension Calculation ---
        # 1. Create a dummy tensor of zeros with a batch size of 1.
        # The asterisk (*) unpacks the tuple into arguments: (1, C, H, W)
        dummy_input = torch.zeros(1, *input_shape)
        
        # 2. Pass the dummy tensor through the feature extractor.
        # We use torch.no_grad() because we don't need to track gradients for this step.
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
            
        # 3. Calculate the total number of elements for a single sample.
        # .view(-1) flattens the tensor into 1D, and .shape[0] gets the length.
        self.flattened_size = dummy_output.view(-1).shape[0]
        
        # --- Classification Layers ---
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(self.flattened_size, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(self.flattened_size, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(16, num_actions)
        )

    def forward(self, x):
        # The forward pass is now incredibly simple
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x