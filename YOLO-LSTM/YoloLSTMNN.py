import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MarioYOLOLSTMNN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=128, lstm_layers=2, dropout=0.1):
        """
        input_shape: tuple of (num_frames, num_objects)
        The model expects inputs of shape (batch_size, num_frames, num_objects).
        """
        super(MarioYOLOLSTMNN, self).__init__()

        assert len(input_shape) == 2, "input_shape must be (num_frames, num_objects)"
        self.num_frames, self.num_objects = input_shape
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # --- Sequence model ---
        self.lstm = nn.LSTM(
            input_size=self.num_objects,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout if self.lstm_layers > 1 else 0.0,
        )

        # --- Fully connected head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        # x shape should be (batch_size, num_frames, num_objects)
        # The model returns logits for each action.
        output, _ = self.lstm(x)
        final_step = output[:, -1, :]
        return self.classifier(final_step)
