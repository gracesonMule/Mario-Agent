import torch
import torch.nn as nn
from torch.distributions import Categorical


class MarioYOLOLSTMActorCritic(nn.Module):
    """
    Actor-Critic network for PPO built on the same LSTM + tracking backbone.

    Architecture:
        Input (B, T, D)
          -> reshape to (B, T, MAX_OBJECTS, FEATURES_PER_OBJ)
          -> compute tracking augmentation (Mario, enemies, coins/powerups)
          -> project back to original feature dim
          -> flatten to (B, T, D)
          -> LSTM -> last hidden state
          -> shared hidden layer
          -> actor head  (policy logits)
          -> critic head (state value scalar)
    """

    def __init__(self, input_shape, num_actions, hidden_size=128, lstm_layers=2, dropout=0.1):
        super().__init__()

        assert len(input_shape) == 2, "input_shape must be (num_frames, feature_dim)"
        self.num_frames, self.feature_dim = input_shape
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_actions = num_actions

        # --- Constants matching Training_Loop preprocessing ---
        self.NUM_YOLO_CLASSES = 37
        self.MAX_OBJECTS = 30
        self.FEATURES_PER_OBJ = self.NUM_YOLO_CLASSES + 4

        # --- Tracking projector ---
        self.track_projector = nn.Linear(self.FEATURES_PER_OBJ + 1, self.FEATURES_PER_OBJ)

        # --- Sequence model ---
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout if self.lstm_layers > 1 else 0.0,
        )

        # --- Shared trunk after LSTM ---
        self.shared = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # --- Actor head (policy) ---
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        # --- Critic head (value) ---
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Orthogonal init (standard for PPO — helps early training stability)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization with small gain on the actor output layer
        so the initial policy is near-uniform across actions."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for module in [self.shared, self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Small gain on the final actor layer to start near-uniform
                    gain = 0.01 if (module is self.actor and layer is module[-1]) else 1.0
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)
        # Track projector
        nn.init.orthogonal_(self.track_projector.weight)
        nn.init.zeros_(self.track_projector.bias)

    # -----------------------------------------------------------------
    # Tracking augmentation
    # -----------------------------------------------------------------
    def _augment_with_tracks(self, x):
        """Reshape, compute tracking scalars, project back. Returns (B, T, D)."""
        B, T, D = x.shape

        if D != self.MAX_OBJECTS * self.FEATURES_PER_OBJ:
            return x  # Shape mismatch — skip tracking

        per_obj = x.view(B, T, self.MAX_OBJECTS, self.FEATURES_PER_OBJ).clone()
        tracks = torch.zeros((B, T, self.MAX_OBJECTS, 1), device=x.device, dtype=x.dtype)

        class_slice = slice(0, self.NUM_YOLO_CLASSES)
        x_idx = self.NUM_YOLO_CLASSES
        y_idx = self.NUM_YOLO_CLASSES + 1

        MARIO_ID = 24
        COIN_IDS = {9}
        POWER_IDS = {11, 35, 36}
        ENEMY_IDS = {1, 2, 3, 5, 7, 8, 13, 14, 15, 16, 20, 21, 23, 31, 32}

        for b in range(B):
            for t in range(T):
                objs = per_obj[b, t]
                class_scores, class_idx = objs[:, class_slice].max(dim=1)
                valid_mask = class_scores > 0.1

                mario_mask = (class_idx == MARIO_ID) & valid_mask
                mario_pos = None
                if mario_mask.any():
                    mi = torch.nonzero(mario_mask, as_tuple=False)[0].item()
                    mario_pos = (objs[mi, x_idx].item(), objs[mi, y_idx].item())
                    tracks[b, t, mi, 0] = 1.0

                if mario_pos is None:
                    continue

                cx = objs[:, x_idx]
                cy = objs[:, y_idx]
                dists = torch.sqrt((cx - mario_pos[0]) ** 2 + (cy - mario_pos[1]) ** 2)

                cp_indices = [i for i in range(self.MAX_OBJECTS)
                              if valid_mask[i] and int(class_idx[i].item()) in (COIN_IDS | POWER_IDS)]
                if cp_indices:
                    for rank, idx in enumerate(sorted(cp_indices, key=lambda i: float(dists[i]))[:3]):
                        tracks[b, t, idx, 0] = 2.0 + rank

                en_indices = [i for i in range(self.MAX_OBJECTS)
                              if valid_mask[i] and int(class_idx[i].item()) in ENEMY_IDS]
                if en_indices:
                    for rank, idx in enumerate(sorted(en_indices, key=lambda i: float(dists[i]))[:3]):
                        tracks[b, t, idx, 0] = 5.0 + rank

        tracks = tracks / 7.0
        augmented = torch.cat([per_obj, tracks], dim=-1)
        projected = self.track_projector(augmented)
        return projected.view(B, T, -1)

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (B, T, D) — stacked frame feature vectors
        Returns:
            logits: (B, num_actions) — raw policy logits
            value:  (B, 1)          — state value estimate
        """
        x = self._augment_with_tracks(x)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]       # last timestep hidden state
        shared_out = self.shared(h)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value

    def get_action_and_value(self, x, action=None):
        """
        Convenience method used by PPO:
          - During rollout (action=None): sample an action, return it + log_prob + value
          - During update  (action given): compute log_prob and entropy for that action
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
