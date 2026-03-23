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
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import MarioCNN

MOVING_AVG = 10000
EPISODES = 5_000_000
LR = 0.0001
LR_DECAY = 0.99

STARTING_EPSILON = 0.1 
MIN_EPSILON = 0.005 # Smallest possible value for Epsilon
EPSILON_DECAY = 0.99999975

# --- TRAINING LEVELS CONFIGURATION ---
# Easily modify this list to train on different levels
TRAINING_LEVELS = [
    'SuperMarioBros-1-1-v0',
    'SuperMarioBros-1-2-v0',
    'SuperMarioBros-2-1-v0',
    'SuperMarioBros-2-2-v0',
    'SuperMarioBros-2-4-v0',
    'SuperMarioBros-4-3-v0',
    'SuperMarioBros-8-2-v0',
    'SuperMarioBros-8-3-v0'
]
CHECKPOINT_INTERVAL = 10000  # Report stats every N epochs

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        
        # A deque automatically pushes out the oldest frame when a new one is added
        self.frames = deque(maxlen=num_frames)
        
        # We need to update the observation space so the agent knows what to expect
        # It changes from (84, 84, 1) to (84, 84, 4)
        old_space = env.observation_space
        self.observation_space = Box(
            low=np.repeat(old_space.low, num_frames, axis=-1),
            high=np.repeat(old_space.high, num_frames, axis=-1),
            dtype=old_space.dtype
        )

    def reset(self):
        """When the game resets, we fill the stack with 4 copies of the starting frame."""
        obs = self.env.reset()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        """Every time we take a step, add the new frame to the stack."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Concatenates our 4 separate (84, 84, 1) frames into a single (84, 84, 4) block."""
        return np.concatenate(list(self.frames), axis=-1)

class GrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        
        # Update the environment's observation space to expect an 84x84 image with 1 color channel (grayscale)
        self.observation_space = Box(
            low=0, 
            high=255, 
            shape=(self.shape[0], self.shape[1], 1), 
            dtype=np.uint8
        )

    def observation(self, obs):
        """
        This method automatically intercepts the frame every time env.step() is called.
        """
        # 1. Convert from RGB to Grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize the image to 84x84
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        
        # 3. Add the channel dimension back so the shape is (84, 84, 1)
        # Neural networks generally expect that channel dimension!
        final_obs = np.expand_dims(resized, axis=-1)
        
        return final_obs

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

        # It needs to know it is receiving 4 stacked frames and outputting 7 possible actions
        self.net = MarioCNN.MarioCNN(input_shape=(4, 84, 84), num_actions=action_space_size).to(self.device)

        # target network (frozen judge)
        self.target_net = MarioCNN.MarioCNN(input_shape=(4, 84, 84), num_actions=action_space_size).to(self.device)

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

        # Epsilon-Greedy parameters
        self.exploration_rate = STARTING_EPSILON
        self.exploration_rate_min = MIN_EPSILON
        self.exploration_rate_decay = EPSILON_DECAY

        # --- Sync Tracker ---
        self.learn_step_counter = 0      # Tracks how many times learn() is called
        self.sync_every = 10000          # How often to copy weights to the target network

    def sync_target_network(self):
        """Copies the weights from the online network to the target network."""
        self.target_net.load_state_dict(self.net.state_dict())
        print(f"Target Network Synced at step {self.learn_step_counter}!")

    def learn(self, states, actions, rewards, next_states, dones, weights):
        states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        
        states = states.to(self.device)
        next_states = next_states.to(self.device)
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
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()

        # --- Weighted Backpropagation ---
        # Change SmoothL1Loss to not reduce automatically
        loss_fn = nn.SmoothL1Loss(reduction='none') 
        elementwise_loss = loss_fn(current_q, target_q)

        # Apply the Importance Sampling weights
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
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
            # EXPLOIT: Use CNN
            state_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            state_tensor = state_tensor.to(self.device)
            
            self.net.eval() # Turn off Dropout for predicting
            with torch.no_grad():
                action_values = self.net(state_tensor)
            action_idx = torch.argmax(action_values, dim=1).item()
            self.net.train() # Turn Dropout back on for learning

        # Decay Epsilon
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

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
            plt.plot(epochs, moving_avg, color=colors[idx], label=f'{level_name} (MA)', linewidth=1.5, alpha=0.8)
    
    # Calculate and plot overall moving average
    all_scores = []
    for level_name in TRAINING_LEVELS:
        all_scores.extend(level_scores[level_name])
    
    all_scores = np.array(all_scores)
    if len(all_scores) >= ma:
        overall_ma = np.convolve(all_scores, np.ones(ma)/ma, mode='valid')
        epochs = range(ma - 1, len(all_scores))
        plt.plot(epochs, overall_ma, color='black', label='Overall Average (MA)', linewidth=3, alpha=0.9)
    
    plt.title("Mario Agent Training Progress - Moving Averages by Level")
    plt.xlabel(f"Epoch (MA Window: {ma})")
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

    model_outdir = os.path.join(outdir, "mario_cnn_model.pth")
    weights_outdir = os.path.join(outdir, "mario_cnn_weights.pth")

    # --- Create a wrapper function to initialize level environments ---
    def create_level_env(level_name):
        env = gym_super_mario_bros.make(level_name)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = StuckPenaltyWrapper(env, max_steps_stuck=25, penalty=-15.0)
        env = GrayScaleResizeWrapper(env, shape=(84, 84))
        env = FrameStackWrapper(env, num_frames=4)
        return env

    # Initialize environments for all levels
    print(f"Initializing {len(TRAINING_LEVELS)} training levels...")
    level_envs = {level: create_level_env(level) for level in TRAINING_LEVELS}
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
            
            print(f"Epoch: {epoch_count:7d} | Level: {level_name:30s} | Score: {total_reward:8.1f} | Epsilon: {agent.exploration_rate:.6f}")

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