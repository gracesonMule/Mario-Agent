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

        # Constants describing how the flattened vector is organized.
        # These must match the preprocessing in Training_Loop.yolo_to_lstm_vector
        self.NUM_YOLO_CLASSES = 37
        self.MAX_OBJECTS = 30
        self.FEATURES_PER_OBJ = self.NUM_YOLO_CLASSES + 4

        # --- Sequence model ---
        # The LSTM will consume a flattened per-frame vector (MAX_OBJECTS * FEATURES_PER_OBJ)
        self.lstm = nn.LSTM(
            input_size=self.num_objects,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout if self.lstm_layers > 1 else 0.0,
        )

        # A small learned projector that allows us to augment each object's features
        # with a single "track" scalar and then compress back to the original
        # per-object feature size so the external input shape does not change.
        self.track_projector = nn.Linear(self.FEATURES_PER_OBJ + 1, self.FEATURES_PER_OBJ)

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

        # Safety: if the provided per-frame vector equals the expected flattened size,
        # reshape to (B, T, MAX_OBJECTS, FEATURES_PER_OBJ) to compute tracks.
        B, T, D = x.shape

        # Only attempt tracking if the shape is compatible with expected organization.
        if D == (self.MAX_OBJECTS * self.FEATURES_PER_OBJ):
            # Reshape into object slots
            per_obj = x.view(B, T, self.MAX_OBJECTS, self.FEATURES_PER_OBJ).clone()

            # Prepare track tensor (0 = no track). We'll assign 1..7, then normalize by 7.
            tracks = torch.zeros((B, T, self.MAX_OBJECTS, 1), device=x.device, dtype=x.dtype)

            # Class one-hot region and bbox indices
            class_slice = slice(0, self.NUM_YOLO_CLASSES)
            x_idx = self.NUM_YOLO_CLASSES
            y_idx = self.NUM_YOLO_CLASSES + 1

            # Define ids (from synth-data.yaml)
            MARIO_ID = 24
            COIN_IDS = {9}
            POWER_IDS = {11, 35, 36}
            # Enemy IDs gathered from synth-data.yaml (common enemies)
            ENEMY_IDS = {1, 2, 3, 5, 7, 8, 13, 14, 15, 16, 20, 21, 23, 31, 32}

            # Loop over batch/time - small loops are acceptable here
            for b in range(B):
                for t in range(T):
                    objs = per_obj[b, t]

                    # Determine which slots are valid detections (max class score > 0)
                    class_scores, class_idx = objs[:, class_slice].max(dim=1)
                    valid_mask = class_scores > 0.1

                    # Find mario index (first valid object with class==MARIO_ID)
                    mario_mask = (class_idx == MARIO_ID) & valid_mask
                    mario_pos = None
                    if mario_mask.any():
                        mi = torch.nonzero(mario_mask, as_tuple=False)[0].item()
                        mx = objs[mi, x_idx].item()
                        my = objs[mi, y_idx].item()
                        mario_pos = (mx, my)
                        tracks[b, t, mi, 0] = 1.0

                    if mario_pos is None:
                        # No Mario detected; skip proximity-based tracks for this frame
                        continue

                    # Compute distances for all valid objects
                    cx = objs[:, x_idx]
                    cy = objs[:, y_idx]
                    dists = torch.sqrt((cx - mario_pos[0]) ** 2 + (cy - mario_pos[1]) ** 2)

                    # COIN/POWER combined set
                    coin_power_mask = valid_mask & ((class_idx.unsqueeze(1) == torch.tensor(list(COIN_IDS.union(POWER_IDS)), device=x.device).unsqueeze(0)).any(dim=1))
                    # But above logic is a bit awkward; compute with python set
                    cp_indices = [i for i in range(self.MAX_OBJECTS) if valid_mask[i] and int(class_idx[i].item()) in (COIN_IDS | POWER_IDS)]
                    if len(cp_indices) > 0:
                        cp_sorted = sorted(cp_indices, key=lambda i: float(dists[i]))[:3]
                        # Assign tracks 2..4: 2 = closest, 4 = furthest
                        for rank, idx in enumerate(cp_sorted):
                            tracks[b, t, idx, 0] = 2.0 + rank

                    # ENEMIES: select up to 3 closest
                    en_indices = [i for i in range(self.MAX_OBJECTS) if valid_mask[i] and int(class_idx[i].item()) in ENEMY_IDS]
                    if len(en_indices) > 0:
                        en_sorted = sorted(en_indices, key=lambda i: float(dists[i]))[:3]
                        # Assign tracks 5..7: 5 = closest, 7 = furthest
                        for rank, idx in enumerate(en_sorted):
                            tracks[b, t, idx, 0] = 5.0 + rank

            # Normalize track values to [0, 1] by dividing by 7
            tracks = tracks / 7.0

            # Concatenate and project back to original per-object feature size
            augmented = torch.cat([per_obj, tracks], dim=-1)
            projected = self.track_projector(augmented)

            # Re-flatten to original D
            x_proc = projected.view(B, T, -1)
        else:
            # Shape mismatch; skip tracking and use input as-is
            x_proc = x

        output, _ = self.lstm(x_proc)
        final_step = output[:, -1, :]
        return self.classifier(final_step)
