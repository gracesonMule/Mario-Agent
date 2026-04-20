"""
CleanRL-style PPO training loop for Mario YOLO-LSTM agent.
Compatible with Python 3.8 and PyTorch 2.0.1+cu118.

Key differences from the old DDQN loop:
  - On-policy: no replay buffer. We collect a fixed-length rollout, then update.
  - No epsilon-greedy: exploration comes from the stochastic policy itself.
  - No target network: PPO uses a clipped surrogate objective instead.
  - Generalized Advantage Estimation (GAE) for variance-reduced advantage computation.
  - Random level sampling per episode for cross-level generalization.
"""

import numpy as np
import cv2
from collections import deque
import argparse
import matplotlib.pyplot as plt
import os
import traceback
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import gym
from gym.spaces import Box
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from YoloLSTMNN import MarioYOLOLSTMActorCritic
from ultralytics import YOLO


# ===========================================================================
# YOLO / Observation Constants
# ===========================================================================
TRAINED_YOLO_MODEL = "train6"
NUM_FRAMES_FOR_YOLO = 4

NUM_YOLO_CLASSES = 37
MAX_OBJECTS = 30
FEATURES_PER_OBJ = NUM_YOLO_CLASSES + 4
INPUT_SIZE = MAX_OBJECTS * FEATURES_PER_OBJ

# ===========================================================================
# PPO Hyperparameters (CleanRL defaults tuned for game environments)
# ===========================================================================
TOTAL_TIMESTEPS = 10_000_000   # Total environment steps across all training
LR = 2.5e-4                    # Learning rate (CleanRL default for Atari)
NUM_STEPS = 512                # Steps per rollout before each PPO update
NUM_MINIBATCHES = 8            # Minibatches per PPO epoch
UPDATE_EPOCHS = 4              # Number of PPO epochs per rollout
GAMMA = 0.99                   # Discount factor
GAE_LAMBDA = 0.95              # GAE lambda for advantage estimation
CLIP_EPS = 0.2                 # PPO clipping parameter
CLIP_VLOSS = True              # Whether to clip the value loss
ENT_COEF = 0.01                # Entropy bonus coefficient (encourages exploration)
VF_COEF = 0.25                 # Value function loss coefficient (reduced to prevent critic dominating shared gradients)
MAX_GRAD_NORM = 0.5            # Gradient clipping
REWARD_CLIP = 5.0              # Clamp rewards to [-REWARD_CLIP, REWARD_CLIP] (always applied as safety net)
RETURN_NORM_BURN_IN = 2048     # Return normalizer collects this many samples before activating
ANNEAL_LR = True               # Linearly anneal LR to 0 over training
NORM_ADV = True                # Normalize advantages within each minibatch

# ===========================================================================
# Training Configuration
# ===========================================================================
TRAINING_LEVELS = [
    'SuperMarioBros-1-1-v0',
    'SuperMarioBros-1-2-v0',
    'SuperMarioBros-2-1-v0',
    'SuperMarioBros-2-2-v0',
    'SuperMarioBros-2-4-v0',
    'SuperMarioBros-4-3-v0',
    'SuperMarioBros-8-2-v0',
    'SuperMarioBros-8-3-v0',
]

def get_available_levels(global_step):
    """Return the list of levels available for training at the current global_step."""
    base_level = ['SuperMarioBros-1-1-v0']
    level_additions = [
        ('SuperMarioBros-1-2-v0', 500000),
        ('SuperMarioBros-2-1-v0', 1000000),
        ('SuperMarioBros-2-2-v0', 1500000),
        ('SuperMarioBros-2-4-v0', 2000000),
        ('SuperMarioBros-4-3-v0', 2500000),
        ('SuperMarioBros-8-2-v0', 3000000),
        ('SuperMarioBros-8-3-v0', 3500000),
    ]
    available = base_level.copy()
    for level, threshold in level_additions:
        if global_step >= threshold:
            available.append(level)
    return available

CHECKPOINT_INTERVAL = 50       # Episodes between stat reports
SAVE_INTERVAL = 500            # Episodes between model saves
MOVING_AVG = 100               # Window for progress plot


# ===========================================================================
# YOLO Preprocessing (unchanged from your code)
# ===========================================================================
def yolo_to_lstm_vector(results):
    """Converts a YOLO Results object into a fixed-size 1D numpy array."""
    frame_features = np.zeros((MAX_OBJECTS, FEATURES_PER_OBJ), dtype=np.float32)
    boxes = results[0].boxes
    num_detected = min(len(boxes), MAX_OBJECTS)

    if num_detected > 0:
        classes = boxes.cls[:num_detected].cpu().numpy().astype(int)
        coords = boxes.xywh[:num_detected].cpu().numpy()

        for i in range(num_detected):
            one_hot = np.zeros(NUM_YOLO_CLASSES, dtype=np.float32)
            cls_id = classes[i]
            if 0 <= cls_id < NUM_YOLO_CLASSES:
                one_hot[cls_id] = 1.0

            x_norm = coords[i][0] / 256.0
            y_norm = coords[i][1] / 240.0
            w_norm = coords[i][2] / 256.0
            h_norm = coords[i][3] / 240.0

            frame_features[i] = np.concatenate([one_hot, [x_norm, y_norm, w_norm, h_norm]])

    return frame_features.flatten()


# ===========================================================================
# Environment Wrappers (unchanged from your code)
# ===========================================================================
class StuckPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, max_steps_stuck=25, penalty=-15.0):
        super().__init__(env)
        self.max_steps_stuck = max_steps_stuck
        self.penalty = penalty
        self.x_pos_history = deque(maxlen=max_steps_stuck)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.x_pos_history.clear()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_x = info.get('x_pos', 0)
        self.x_pos_history.append(current_x)
        if len(self.x_pos_history) == self.max_steps_stuck:
            if max(self.x_pos_history) - min(self.x_pos_history) < 2:
                reward += self.penalty
                done = True
        return obs, reward, done, info


class YoloObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, yolo_model):
        super().__init__(env)
        self.yolo_model = yolo_model
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(INPUT_SIZE,), dtype=np.float32
        )

    def observation(self, obs):
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        masked_input = frame_bgr.copy()
        masked_input[0:31, :] = (0, 0, 0)
        masked_input[224:240, :] = (0, 0, 0)
        results = self.yolo_model(masked_input, verbose=False)
        return yolo_to_lstm_vector(results)


class VectorFrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames=NUM_FRAMES_FOR_YOLO):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        old_space = env.observation_space
        self.observation_space = Box(
            low=np.repeat(old_space.low[np.newaxis, ...], num_frames, axis=0),
            high=np.repeat(old_space.high[np.newaxis, ...], num_frames, axis=0),
            dtype=old_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.stack(list(self.frames), axis=0)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


# ===========================================================================
# Return Normalization with Burn-in
# ===========================================================================
class ReturnNormalizer:
    """Normalizes value function targets (returns) using running statistics.

    The critic learns to predict in *normalized* space where targets hover
    near 0 with unit variance.  During GAE, the critic's predictions are
    denormalized back to the original reward scale so that TD errors are
    computed correctly.

    Uses a burn-in pattern: during burn-in the
    normalizer is transparent (passes values through unchanged) while it
    accumulates a stable variance estimate.
    """

    def __init__(self, burn_in=2048):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.burn_in = burn_in

    @property
    def is_warmed_up(self):
        return self.count >= self.burn_in

    @property
    def std(self):
        return max(self.var ** 0.5, 1e-8)

    def update(self, returns_tensor):
        """Update running stats from a batch of returns (tensor or numpy)."""
        if isinstance(returns_tensor, torch.Tensor):
            vals = returns_tensor.detach().cpu().numpy().flatten()
        else:
            vals = np.asarray(returns_tensor).flatten()

        for v in vals:
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x):
        """Normalize returns for use as critic targets.
        Before burn-in: returns pass through unchanged.
        After  burn-in: (x - mean) / std
        """
        if not self.is_warmed_up:
            return x
        return (x - self.mean) / self.std

    def denormalize(self, x):
        """Convert critic predictions back to original reward scale for GAE.
        Before burn-in: values pass through unchanged.
        After  burn-in: x * std + mean
        """
        if not self.is_warmed_up:
            return x
        return x * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


# ===========================================================================
# Progress Plotting
# ===========================================================================
def save_progress_plot(level_scores, ma, filename="mario_training_progress.png"):
    plt.figure(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TRAINING_LEVELS)))

    for idx, level_name in enumerate(TRAINING_LEVELS):
        data = level_scores.get(level_name, [])
        if len(data) == 0:
            continue
        episodes, scores = zip(*data)
        episodes = np.array(episodes)
        scores = np.array(scores)
        plt.plot(episodes, scores, color=colors[idx], linewidth=1, alpha=0.3)
        if len(scores) >= ma:
            moving_avg = np.convolve(scores, np.ones(ma) / ma, mode='valid')
            ma_episodes = episodes[ma-1:]
            plt.plot(ma_episodes, moving_avg, color=colors[idx], label=f'{level_name}', linewidth=2.5, alpha=0.9)

    plt.title("Mario PPO Training Progress")
    plt.xlabel(f"Global Episode (MA Window: {ma})")
    plt.ylabel("Total Reward")
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


# ===========================================================================
# PPO Training
# ===========================================================================
def main(model_path, outdir):
    seed = 486
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA\n")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS\n")
    else:
        device = torch.device("cpu")
        print("Using CPU\n")

    # --- YOLO ---
    yolo_model = YOLO(f"runs/detect/{TRAINED_YOLO_MODEL}/weights/best.pt")

    # --- Environment factory ---
    def create_level_env(level_name):
        env = gym_super_mario_bros.make(level_name)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = StuckPenaltyWrapper(env, max_steps_stuck=25, penalty=-15.0)
        env = YoloObservationWrapper(env, yolo_model=yolo_model)
        env = VectorFrameStackWrapper(env, num_frames=NUM_FRAMES_FOR_YOLO)
        return env

    print(f"Initializing {len(TRAINING_LEVELS)} training levels...")
    level_envs = {level: create_level_env(level) for level in TRAINING_LEVELS}
    print(f"Levels initialized: {TRAINING_LEVELS}\n")

    # --- Get shapes from any env ---
    sample_env = level_envs[TRAINING_LEVELS[0]]
    obs_shape = sample_env.observation_space.shape   # (NUM_FRAMES, INPUT_SIZE)
    num_actions = sample_env.action_space.n

    # --- Network ---
    agent = MarioYOLOLSTMActorCritic(
        input_shape=(NUM_FRAMES_FOR_YOLO, INPUT_SIZE),
        num_actions=num_actions,
    ).to(device)

    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, nn.Module):
            agent.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            agent.load_state_dict(checkpoint)
        print(f"Loaded weights from {model_path}")

    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # --- Restore optimizer state and training counters if resuming ---
    episode_count = 0
    global_step = 0
    level_scores = {level: [] for level in TRAINING_LEVELS}

    if model_path and isinstance(checkpoint, dict):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Restored optimizer state (Adam momentum/variance)")
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            print(f"  Resuming from global_step {global_step:,}")
        if 'episode_count' in checkpoint:
            episode_count = checkpoint['episode_count']
            print(f"  Resuming from episode {episode_count}")
        if 'level_scores' in checkpoint:
            saved_scores = checkpoint['level_scores']
            # Merge: keep history for levels that are still in TRAINING_LEVELS,
            # ignore old levels no longer trained, init new levels as empty
            for level in TRAINING_LEVELS:
                if level in saved_scores:
                    level_scores[level] = saved_scores[level]
            print(f"  Restored per-level score history")
        print()

    # --- Rollout storage (pre-allocated tensors for NUM_STEPS) ---
    obs_buf = torch.zeros((NUM_STEPS, *obs_shape), dtype=torch.float32).to(device)
    act_buf = torch.zeros(NUM_STEPS, dtype=torch.long).to(device)
    logprob_buf = torch.zeros(NUM_STEPS, dtype=torch.float32).to(device)
    reward_buf = torch.zeros(NUM_STEPS, dtype=torch.float32).to(device)
    done_buf = torch.zeros(NUM_STEPS, dtype=torch.float32).to(device)
    value_buf = torch.zeros(NUM_STEPS, dtype=torch.float32).to(device)

    # --- Derived schedule values ---
    num_updates = TOTAL_TIMESTEPS // NUM_STEPS
    minibatch_size = NUM_STEPS // NUM_MINIBATCHES
    start_update = (global_step // NUM_STEPS) + 1  # Resume from correct update index
    return_normalizer = ReturnNormalizer(burn_in=RETURN_NORM_BURN_IN)

    if model_path and isinstance(checkpoint, dict):
        if 'return_normalizer' in checkpoint:
            return_normalizer.load_state_dict(checkpoint['return_normalizer'])
            print(f"  Restored return normalizer (count={return_normalizer.count:.0f}, "
                  f"warmed_up={return_normalizer.is_warmed_up})")

    # --- Begin with a random level ---
    current_level = random.choice(get_available_levels(global_step))
    env = level_envs[current_level]
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    episode_reward = 0.0
    done = False

    print(f"PPO Training | {TOTAL_TIMESTEPS:,} total steps | {num_updates} updates")
    print(f"Rollout: {NUM_STEPS} steps | {NUM_MINIBATCHES} minibatches | {UPDATE_EPOCHS} epochs")
    if start_update > 1:
        print(f"Resuming from update {start_update} (step {global_step:,})")
    print()

    start_time = time.time()
    steps_at_session_start = global_step  # For accurate sps calculation on resume

    try:
        for update in range(start_update, num_updates + 1):

            # --- Learning rate annealing (based on absolute position in training) ---
            if ANNEAL_LR:
                frac = 1.0 - (update - 1) / num_updates
                optimizer.param_groups[0]['lr'] = LR * frac

            # ===============================================================
            # PHASE 1: Collect NUM_STEPS of experience (rollout)
            # ===============================================================
            agent.eval()
            for step in range(NUM_STEPS):
                global_step += 1
                obs_buf[step] = obs_tensor
                done_buf[step] = float(done)

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_tensor.unsqueeze(0))

                act_buf[step] = action
                logprob_buf[step] = logprob
                value_buf[step] = value

                obs, reward, done, info = env.step(action.item())

                # Safety clamp only — return normalizer handles scaling at the target level.
                # Do NOT normalize per-step rewards; stacking reward + return normalization
                # crushes variance and makes the critic trivially accurate but useless.
                reward_buf[step] = max(-REWARD_CLIP, min(REWARD_CLIP, reward))

                episode_reward += reward

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

                if done:
                    # Log completed episode
                    level_scores[current_level].append((episode_count, episode_reward))
                    episode_count += 1

                    if episode_count % 10 == 0:
                        print(
                            f"  ep {episode_count:6d} | {current_level:30s} | "
                            f"reward {episode_reward:8.1f} | step {global_step:,}"
                        )

                    # --- Checkpoint reporting ---
                    if episode_count % CHECKPOINT_INTERVAL == 0:
                        elapsed = time.time() - start_time
                        sps = (global_step - steps_at_session_start) / max(elapsed, 1e-8)
                        print("\n" + "=" * 100)
                        print(f"CHECKPOINT @ episode {episode_count} | step {global_step:,} | {sps:.0f} steps/sec")
                        print("=" * 100)
                        for ln in TRAINING_LEVELS:
                            sc = level_scores[ln]
                            if len(sc) > 0:
                                recent_rewards = [r for _, r in sc[-CHECKPOINT_INTERVAL:]]
                                print(f"  {ln:30s} | n={len(sc):5d} | Avg: {np.mean(recent_rewards):8.2f} | Max: {np.max(recent_rewards):8.2f}")
                        print("=" * 100 + "\n")

                        try:
                            plot_path = os.path.join(outdir, "mario_training_progress.png")
                            save_progress_plot(level_scores, ma=MOVING_AVG, filename=plot_path)
                        except Exception as e:
                            print(f"Plot error: {e}")

                    # --- Save model ---
                    if episode_count % SAVE_INTERVAL == 0:
                        save_path = os.path.join(outdir, "mario_ppo_latest.pth")
                        torch.save({
                            'model_state_dict': agent.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'episode_count': episode_count,
                            'level_scores': level_scores,
                            'return_normalizer': return_normalizer.state_dict(),
                        }, save_path)
                        print(f"--> Checkpoint saved at episode {episode_count}")

                    # --- Reset into a randomly chosen level ---
                    current_level = random.choice(get_available_levels(global_step))
                    env = level_envs[current_level]
                    obs = env.reset()
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                    episode_reward = 0.0

            # ===============================================================
            # PHASE 2: Compute advantages via GAE
            # ===============================================================
            with torch.no_grad():
                # Bootstrap value for the last state
                _, _, _, next_value = agent.get_action_and_value(obs_tensor.unsqueeze(0))

                # Denormalize all critic predictions back to the original reward
                # scale so that TD errors mix rewards and values in the same units.
                # Before burn-in this is a no-op (passthrough).
                denorm_values = return_normalizer.denormalize(value_buf)
                denorm_next_value = return_normalizer.denormalize(next_value)

                advantages = torch.zeros_like(reward_buf).to(device)
                lastgaelam = 0.0

                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        next_non_terminal = 1.0 - float(done)
                        next_val = denorm_next_value
                    else:
                        next_non_terminal = 1.0 - done_buf[t + 1]
                        next_val = denorm_values[t + 1]

                    delta = reward_buf[t] + GAMMA * next_val * next_non_terminal - denorm_values[t]
                    lastgaelam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * lastgaelam
                    advantages[t] = lastgaelam

                # Raw returns in the original reward scale
                raw_returns = advantages + denorm_values

                # Update the return normalizer with this batch of returns
                return_normalizer.update(raw_returns)

                # Normalize returns for critic targets (after burn-in)
                returns = return_normalizer.normalize(raw_returns)

            # ===============================================================
            # PHASE 3: PPO optimization (multiple epochs over the rollout)
            # ===============================================================
            agent.train()

            # Flatten rollout into a batch
            b_obs = obs_buf.reshape(-1, *obs_shape)
            b_actions = act_buf.reshape(-1)
            b_logprobs = logprob_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)

            # Indices for shuffled minibatch iteration
            b_inds = np.arange(NUM_STEPS)

            clipfracs = []

            for epoch in range(UPDATE_EPOCHS):
                np.random.shuffle(b_inds)

                for start in range(0, NUM_STEPS, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    log_ratio = new_logprob - b_logprobs[mb_inds]
                    ratio = log_ratio.exp()

                    # Useful diagnostic: approximate KL divergence
                    with torch.no_grad():
                        clipfracs.append(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())

                    mb_advantages = b_advantages[mb_inds]
                    if NORM_ADV:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # --- Clipped policy loss ---
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # --- Value loss (optionally clipped) ---
                    if CLIP_VLOSS:
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            new_value - b_values[mb_inds], -CLIP_EPS, CLIP_EPS
                        )
                        v_loss1 = (new_value - b_returns[mb_inds]) ** 2
                        v_loss2 = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                    else:
                        v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    # --- Entropy bonus ---
                    entropy_loss = entropy.mean()

                    # --- Combined loss ---
                    loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            # --- Per-update logging ---
            if update % 10 == 0:
                elapsed = time.time() - start_time
                sps = (global_step - steps_at_session_start) / max(elapsed, 1e-8)
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"update {update:5d}/{num_updates} | step {global_step:>10,} | "
                    f"pg_loss {pg_loss.item():7.4f} | v_loss {v_loss.item():7.4f} | "
                    f"ent {entropy_loss.item():.4f} | clip% {np.mean(clipfracs):.3f} | "
                    f"lr {current_lr:.2e} | sps {sps:.0f}"
                )

    finally:
        # --- Cleanup ---
        print("\nCleaning up environments...")
        for _, env in level_envs.items():
            try:
                env.close()
            except Exception:
                pass

        # Final save
        save_path = os.path.join(outdir, "mario_ppo_final.pth")
        torch.save({
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'episode_count': episode_count,
            'level_scores': level_scores,
            'return_normalizer': return_normalizer.state_dict(),
        }, save_path)
        print(f"Final model saved to {save_path}")
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for Mario YOLO-LSTM agent")
    parser.add_argument('--loadpath', help="Path to load model checkpoint from.", default=None, required=False)
    parser.add_argument('--outdir', help="Directory to save outputs to.", required=True)
    args = parser.parse_args()
    main(model_path=args.loadpath, outdir=args.outdir)