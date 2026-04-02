import numpy as np
import cv2
from collections import deque
import argparse
import matplotlib.pyplot as plt
import os
import traceback
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import random
from collections import deque

import gym
from gym.spaces import Box
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from YoloLSTMNN import MarioYOLOLSTMNN
from ultralytics import YOLO

TRAINED_YOLO_MODEL = "train6"
NUM_FRAMES_FOR_YOLO = 4

MOVING_AVG = 100
EPISODES = 5_000_000
LR = 0.0001
LR_DECAY = 0.99

STARTING_EPSILON = 1.0
MIN_EPSILON = 0.005 # Smallest possible value for Epsilon
EPSILON_DECAY_STEPS = 1_000_000

# --- TRAINING LEVELS CONFIGURATION ---
# Easily modify this list to train on different levels
TRAINING_LEVELS = [
    'SuperMarioBros-1-1-v0',
    # 'SuperMarioBros-1-2-v0',
    # 'SuperMarioBros-2-1-v0',
    # 'SuperMarioBros-2-2-v0',
    # 'SuperMarioBros-2-4-v0',
    # 'SuperMarioBros-4-3-v0',
    # 'SuperMarioBros-8-2-v0',
    # 'SuperMarioBros-8-3-v0'
]
CHECKPOINT_INTERVAL = 2 * len(TRAINING_LEVELS) # Should be a multiple of levels trained on

# Define our fixed tensor dimensions
MAX_OBJECTS = 10         # The maximum number of sprites we will track per frame
FEATURES_PER_OBJ = 5     # [class_id, x_center, y_center, width, height]
INPUT_SIZE = MAX_OBJECTS * FEATURES_PER_OBJ # Total size of our 1D vector (50)



def yolo_to_lstm_vector(results):
    """Converts a YOLO Results object into a fixed-size 1D numpy array."""
    # 1. Initialize an array of pure zeros
    frame_features = np.zeros((MAX_OBJECTS, FEATURES_PER_OBJ), dtype=np.float32)

    # 2. Extract the bounding box data
    boxes = results[0].boxes

    # 3. Determine how many objects to process (cap it at MAX_OBJECTS)
    num_detected = min(len(boxes), MAX_OBJECTS)

    if num_detected > 0:

        # Move tensors to CPU and convert to numpy for easier handling
        classes = boxes.cls[:num_detected].cpu().numpy()

        # Use xywh (center x, center y, width, height) instead of xyxy. 
        # Neural networks usually learn better from center points and scales.
        coords = boxes.xywh[:num_detected].cpu().numpy() 

        # 4. Populate the array with the actual detections
        for i in range(num_detected):

            # Normalizing coordinates between 0 and 1 is highly recommended for LSTMs!
            # Assuming your game window is 256x240 (standard NES resolution)
            x_norm = coords[i][0] / 256.0
            y_norm = coords[i][1] / 240.0
            w_norm = coords[i][2] / 256.0
            h_norm = coords[i][3] / 240.0

            frame_features[i] = [classes[i], x_norm, y_norm, w_norm, h_norm]

    # 5. Flatten the 2D array (10x5) into a 1D vector (size 50)
    return frame_features.flatten()

class StuckPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, max_steps_stuck=25, penalty=-15.0):
        super().__init__(env)
        # How many steps to wait before deciding Mario is stuck
        self.max_steps_stuck = max_steps_stuck
        # The massive negative reward to teach him a lesson
        self.penalty = penalty
        # A memory queue to track his recent x-coordinates
        self.x_pos_history = deque(maxlen=max_steps_stuck)

    def reset(self, **kwargs):
        """Clear the history every time a new episode starts."""
        obs = self.env.reset(**kwargs)
        self.x_pos_history.clear()
        return obs

    def step(self, action):
        """Intercept the step to check Mario's progress."""
        obs, reward, done, info = self.env.step(action)
        
        # gym-super-mario-bros passes Mario's exact location in the 'info' dictionary
        current_x = info.get('x_pos', 0)
        self.x_pos_history.append(current_x)
        
        # If the history buffer is full, check if he actually moved
        if len(self.x_pos_history) == self.max_steps_stuck:
            # If the difference between his furthest and closest x-position is less than 2 pixels, he is stuck
            if max(self.x_pos_history) - min(self.x_pos_history) < 2:
                reward += self.penalty
                done = True  # Instantly kill the episode so we don't waste training time!
                
        return obs, reward, done, info

class YoloObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, yolo_model):
        super().__init__(env)
        self.yolo_model = yolo_model
        
        # Update the environment's observation space to expect a 1D vector of size 50
        self.observation_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(INPUT_SIZE,), 
            dtype=np.float32
        )

    def observation(self, obs):
        """Intercept the raw RGB frame, mask UI, pass it to YOLO, and return the 1D vector."""
        
        # 1. Convert to BGR (matching your script's color format for YOLO)
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        
        # 2. Create a copy to mask so we don't alter the underlying environment frame
        masked_input = frame_bgr.copy()
        
        # 3. Apply UI Masking for YOLO
        masked_input[0:31, :] = (0, 0, 0)     # Mask top score/time UI
        masked_input[224:240, :] = (0, 0, 0)  # Mask bottom boundary/UI
        
        # 4. Run YOLO on the masked input
        results = self.yolo_model(masked_input, verbose=False)
        
        # 5. Convert bounding boxes to the 1D vector for the LSTM
        return yolo_to_lstm_vector(results)
    
class VectorFrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames=NUM_FRAMES_FOR_YOLO):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
        # Update space to represent a sequence of vectors: shape (4, 50)
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
        """Stacks the 1D arrays into a 2D array of shape (num_frames, 50)."""
        return np.stack(list(self.frames), axis=0)

class LinearSchedule:
    def __init__(self, start_val, end_val, total_steps):
        """
        Linearly scales a value from start_val to end_val over total_steps.
        """
        self.start_val = start_val
        self.end_val = end_val
        self.total_steps = total_steps
        self.current_step = 0

    def value(self):
        """Returns the current value based on the step count."""
        # Calculate how far along we are (from 0.0 to 1.0)
        fraction = min(float(self.current_step) / self.total_steps, 1.0)
        
        # Linearly interpolate between start and end
        return self.start_val + fraction * (self.end_val - self.start_val)

    def step(self):
        """Increments the internal step counter."""
        self.current_step += 1

class MarioAgent:
    def __init__(self, action_space_size, model_path=None):
        self.action_space_size = action_space_size

        # If you have an Apple Silicon Mac (M1/M2/M3), you can use the MPS chip for speed!
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU...\n")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS...\n")
        else:
            self.device = torch.device("cpu")
            print("Using CPU...\n")

        # Initialize the LSTM network. Input shape is (num_frames, vector_size) -> (NUM_FRAMES_FOR_YOLO, 50)
        self.net = MarioYOLOLSTMNN(input_shape=(NUM_FRAMES_FOR_YOLO, INPUT_SIZE), num_actions=action_space_size).to(self.device)
        self.target_net = MarioYOLOLSTMNN(input_shape=(NUM_FRAMES_FOR_YOLO, INPUT_SIZE), num_actions=action_space_size).to(self.device)

        # Clone the starting weights so they match perfectly
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Load trained weights if you have them
        if model_path:
            warnings.filterwarnings("error")

            # Check if user loaded a full model object
            try:
                weights = torch.load(model_path, map_location=self.device, weights_only=True)
            except UserWarning as e:
                full_model = torch.load(model_path, map_location=self.device, weights_only=False)
                weights = full_model.state_dict()
                
            warnings.resetwarnings()
                
            self.net.load_state_dict(weights)
            self.target_net.load_state_dict(weights)
            print(f"Loaded model weights from {model_path}")
            
        # Learning parameters
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.scheduler = StepLR(self.optimizer, step_size=50000, gamma=LR_DECAY)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = 0.99

        self.epsilon_schedule = LinearSchedule(
            start_val=STARTING_EPSILON,
            end_val=MIN_EPSILON,
            total_steps=EPSILON_DECAY_STEPS
        )

        # --- Sync Tracker ---
        self.learn_step_counter = 0      # Tracks how many times learn() is called
        self.sync_every = 10000          # How often to copy weights to the target network

    @property
    def exploration_rate(self):
        """Current epsilon value, driven by the linear schedule."""
        return self.epsilon_schedule.value()

    def sync_target_network(self):
        """Copies the weights from the online network to the target network."""
        self.target_net.load_state_dict(self.net.state_dict())
        print(f"Target Network Synced at step {self.learn_step_counter}!")

    def learn(self, states, actions, rewards, next_states, dones, weights):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        # What did the ONLINE network predict?
        current_q = self.net(states).gather(1, actions)

        # --- What is the target value based on the TARGET network? ---
        with torch.no_grad():
            # 1. Use the ONLINE network to choose the best action for the next state
            best_actions_next = self.net(next_states).argmax(dim=1, keepdim=True)
            
            # 2. Use the TARGET network to evaluate the Q-value of that specific action
            next_q = self.target_net(next_states).gather(1, best_actions_next)
                        
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # --- Calculate TD Error for Priorities ---
        # We use .detach() to ensure we don't accidentally backpropagate through the priorities
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy().squeeze(axis=1)

        # --- Weighted Backpropagation ---
        # Change SmoothL1Loss to not reduce automatically
        loss_fn = nn.SmoothL1Loss(reduction='none') 
        elementwise_loss = loss_fn(current_q, target_q)

        # Apply the Importance Sampling weights
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.sync_every == 0:
            self.sync_target_network()
            
        # Return the loss and the new priorities (adding epsilon so priority > 0)
        return loss.item(), td_errors + 1e-5

    def act(self, observation):
        """Chooses an action based on Epsilon-Greedy exploration."""
        if np.random.rand() < self.exploration_rate:
            # EXPLORE
            action_idx = np.random.randint(self.action_space_size)
        else:
            # EXPLOIT: Use YOLO-LSTM
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.net.eval()
            with torch.no_grad():
                action_values = self.net(state_tensor)
            action_idx = torch.argmax(action_values, dim=1).item()
            self.net.train()

        return action_idx    
    
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        # Pre-allocate numpy arrays for speed
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition and assigns it the maximum known priority."""
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Samples a batch weighted by priority, returning IS weights and indices."""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]
            
        # Calculate probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        batch = [self.memory[idx] for idx in indices]
        
        # Calculate Importance Sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize for stability
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.bool_), 
                indices, np.array(weights, dtype=np.float32))

    def update_priorities(self, batch_indices, batch_priorities):
        """Updates the priorities for the sampled batch after learning."""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def save_progress_plot(level_scores, ma, filename="mario_training_progress.png"):
    """
    Saves a multi-level moving average plot.
    Each level gets a thin line with a unique color.
    Overall moving average is a thick line.
    """
    plt.figure(figsize=(14, 7))
    
    # Color palette for different levels
    colors = plt.cm.tab10(np.linspace(0, 1, len(TRAINING_LEVELS)))
    
    # Plot moving average for each level
    for idx, level_name in enumerate(TRAINING_LEVELS):
        scores = np.array(level_scores[level_name])
        if len(scores) >= ma:
            moving_avg = np.convolve(scores, np.ones(ma)/ma, mode='valid')
            # Shift to align with epoch numbers
            epochs = range(ma - 1, len(scores))
            plt.plot(epochs, moving_avg, color=colors[idx], label=f'{level_name} (MA)', linewidth=2.5, alpha=0.9)
    
    # Calculate and plot overall moving average per training cycle
    # Build per-cycle averages (average across levels for each cycle index)
    min_len = min(len(level_scores[level_name]) for level_name in TRAINING_LEVELS)
    if min_len >= ma:
        per_cycle_avg = np.array([
            np.mean([level_scores[level_name][i] for level_name in TRAINING_LEVELS])
            for i in range(min_len)
        ])
        overall_ma = np.convolve(per_cycle_avg, np.ones(ma)/ma, mode='valid')
        epochs = range(ma - 1, min_len)
        # Plot overall MA as solid black on top
        plt.plot(epochs, overall_ma, color='black', linestyle='-', label='Overall Average (MA)', linewidth=3.5, alpha=0.95, zorder=100)
    
    plt.title("Mario Agent Training Progress - Moving Averages by Level")
    plt.xlabel(f"Episode (MA Window: {ma})")
    plt.ylabel("Total Reward")
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Save and close
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

def main(model_path, outdir):
    seed = 486
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_outdir = os.path.join(outdir, "mario_lstm_model.pth")
    weights_outdir = os.path.join(outdir, "mario_lstm_weights.pth")

    yolo_model = YOLO(f"runs/detect/{TRAINED_YOLO_MODEL}/weights/best.pt")

    # --- Create a wrapper function to initialize level environments ---
    def create_level_env(level_name, yolo_instance):
        env = gym_super_mario_bros.make(level_name)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = StuckPenaltyWrapper(env, max_steps_stuck=25, penalty=-15.0)
        env = YoloObservationWrapper(env, yolo_model=yolo_instance) # Replaces Grayscale
        env = VectorFrameStackWrapper(env, num_frames=NUM_FRAMES_FOR_YOLO)
        return env

    # Initialize environments for all levels
    print(f"Initializing {len(TRAINING_LEVELS)} training levels...")
    level_envs = {level: create_level_env(level, yolo_model) for level in TRAINING_LEVELS}
    print(f"Levels initialized: {TRAINING_LEVELS}\n")

    # Instantiate your agent (use the action space from the first level - all should be the same)
    agent = MarioAgent(action_space_size=level_envs[TRAINING_LEVELS[0]].action_space.n, model_path=model_path)
    
    # Initialize the memory buffer to hold the last 50,000 steps
    memory = PrioritizedReplayMemory(capacity=50000)
    batch_size = 32
    
    # --- Tracking dictionaries for per-level and overall scores ---
    level_scores = {level: [] for level in TRAINING_LEVELS}
    epoch_count = 0
    global_step = 0
    
    # Step the epsilon schedule on global_step (linear decay)
    agent.epsilon_schedule.step()

    beta_schedule = LinearSchedule(start_val=0.4, end_val=1.0, total_steps=1_000_000)

    episodes = EPISODES
    
    for ep in range(episodes):
        # --- Play through each level in the training set ---
        for level_name in TRAINING_LEVELS:
            env = level_envs[level_name]
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # --- Increment the step counter ---
                global_step += 1
                
                beta_schedule.step()
                # --- Only learn every 4 steps! ---
                if len(memory) >= batch_size and global_step % 4 == 0:
                    current_beta = beta_schedule.value()
                    b_states, b_actions, b_rewards, b_next_states, b_dones, b_indices, b_weights = memory.sample(batch_size, beta=current_beta)
                    
                    loss, td_errors = agent.learn(b_states, b_actions, b_rewards, b_next_states, b_dones, b_weights)
                    
                    memory.update_priorities(b_indices, td_errors)

            # --- Log level score ---
            level_scores[level_name].append(total_reward)
            epoch_count += 1
            
            agent.scheduler.step()

            print(f"Episode: {ep:7d} | Level: {level_name:30s} | Score: {total_reward:8.1f} | Epsilon: {agent.exploration_rate:.6f}")

        # --- Report statistics every CHECKPOINT_INTERVAL epochs ---
        if epoch_count % CHECKPOINT_INTERVAL == 0:
            print("\n" + "=" * 100)
            print(f"CHECKPOINT AT EPOCH {epoch_count}")
            print("=" * 100)
            
            # Calculate and display per-level statistics
            for level_name in TRAINING_LEVELS:
                scores = level_scores[level_name]
                if len(scores) > 0:
                    recent_scores = scores[-CHECKPOINT_INTERVAL:]
                    avg_score = np.mean(recent_scores)
                    max_score = np.max(recent_scores)
                    min_score = np.min(recent_scores)
                    print(f"  {level_name:30s} | Avg: {avg_score:8.2f} | Max: {max_score:8.2f} | Min: {min_score:8.2f}")
            
            # Calculate and display overall average
            all_recent_scores = []
            for level_name in TRAINING_LEVELS:
                all_recent_scores.extend(level_scores[level_name][-CHECKPOINT_INTERVAL:])
            
            if len(all_recent_scores) > 0:
                overall_avg = np.mean(all_recent_scores)
                print(f"\n  {'OVERALL AVERAGE':30s} | Score: {overall_avg:8.2f}")
            
            print("=" * 100 + "\n")
            
            # Save the progress plot
            try:
                file_path = os.path.join(outdir, "mario_training_progress.png")
                save_progress_plot(level_scores, ma=MOVING_AVG, filename=file_path)
                print(f"--> Plot saved to {file_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")
                traceback.print_exc()

        # Every 10000 checkpoints, save the model
        if (epoch_count % (CHECKPOINT_INTERVAL * 10000)) == 0:
            torch.save(agent.net, model_outdir)
            torch.save(agent.net.state_dict(), weights_outdir)
            print(f"--> Model saved at epoch {epoch_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model from path")

    parser.add_argument('--loadpath', help="Path to load model or weights from.", default=None, required=False)
    
    parser.add_argument('--outdir', help="Directory to save model and weights to.", default=None, required=True)

    args = parser.parse_args()

    main(model_path=args.loadpath, outdir=args.outdir)