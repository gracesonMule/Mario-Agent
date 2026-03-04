import numpy as np
import cv2
from collections import deque

import matplotlib.pyplot as plt

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
from gym_super_mario_bros.actions import RIGHT_ONLY

import MarioCNN

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

class MarioAgent:
    def __init__(self, action_space_size, model_path=None):
        self.action_space_size = action_space_size
        
        # 1. Instantiate your CNN
        # It needs to know it is receiving 4 stacked frames and outputting 7 possible actions
        self.net = MarioCNN.MarioCNN(input_shape=(4, 84, 84), num_actions=action_space_size)
        
        # 2. Load trained weights if you have them
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")
            
        
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

        self.net.to(self.device)

        # Learning parameters
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=50000, gamma=0.1)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = 0.99

        # Epsilon-Greedy parameters
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.1
        self.exploration_rate_decay = 0.99999975

    def learn(self, states, actions, rewards, next_states, dones):
        """Trains YOUR CNN on a batch of memories."""
        states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # What did your network predict?
        current_q = self.net(states).gather(1, actions)

        # What is the target value based on the next state?
        with torch.no_grad():
            next_q = self.net(next_states).max(1)[0].unsqueeze(1)
            
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Backpropagation through your network!
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def act(self, observation):
        """Chooses an action based on Epsilon-Greedy exploration."""
        if np.random.rand() < self.exploration_rate:
            # EXPLORE
            action_idx = np.random.randint(self.action_space_size)
        else:
            # EXPLOIT: Use YOUR CNN
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
    
class ReplayMemory:
    def __init__(self, capacity):
        # A deque is a double-ended queue. When it hits 'maxlen', 
        # it automatically drops the oldest item to make room for the new one.
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Saves a single transition to the memory buffer.
        """
        # We store everything as raw data (ints, floats, and numpy arrays) to save RAM.
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly grabs a batch of transitions to train the neural network.
        """
        # 1. Grab 'batch_size' number of random transitions
        batch = random.sample(self.memory, batch_size)
        
        # 2. 'Unzip' the batch. 
        # This turns a list of tuples like [(s1, a1, r1...), (s2, a2, r2...)]
        # into separate lists: states=[s1, s2], actions=[a1, a2], etc.
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 3. Convert them to NumPy arrays for speed
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.bool_)
        
        # 4. Return them so your training loop can convert them to PyTorch tensors
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Allows us to check how full the memory is by calling len(memory)."""
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

def save_progress_plot(rewards, filename="mario_training_progress.png"):
    """Saves a line graph of the agent's rewards over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color='blue', label='Episode Reward')
    
    # Add a trendline (Moving Average of the last 10 episodes)
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        # Shift the moving average to align with the end of the graph
        plt.plot(range(9, len(rewards)), moving_avg, color='orange', label='10-Episode Moving Avg', linewidth=2)
        
    plt.title("Mario Agent Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # Save the file and close the plot so it doesn't eat up RAM
    plt.savefig(filename)
    plt.close()

def main():
    seed = 486
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Initialize the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    # 2. Restrict the action space to standard Mario movements (0 to 6)
    # This makes it much easier for a neural network to learn
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    
    env = StuckPenaltyWrapper(env, max_steps_stuck=25, penalty=-15.0)

    env = GrayScaleResizeWrapper(env, shape=(84, 84))

    env = FrameStackWrapper(env, num_frames=4)

    # 3. Instantiate your agent
    agent = MarioAgent(action_space_size=env.action_space.n)
    
    # 4. Start the game loop, how times to run
    episodes = 500
    
# 1. Initialize the memory buffer to hold the last 50,000 steps
    memory = ReplayMemory(capacity=50000)
    batch_size = 32
    
    episode_rewards = []
    
# --- NEW: Add a global step counter ---
    global_step = 0 
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # --- NEW: Increment the step counter ---
            global_step += 1 
            
            # --- FIXED: Only learn every 4 steps! ---
            if len(memory) >= batch_size and global_step % 4 == 0:
                b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(batch_size)
                loss = agent.learn(b_states, b_actions, b_rewards, b_next_states, b_dones)

        # --- NEW: Logging at the end of every episode ---
        episode_rewards.append(total_reward)
        print(f"Episode: {ep + 1} | Score: {total_reward} | Epsilon: {agent.exploration_rate:.4f}")
        
        # Every 10 episodes, save the model and update the graph
        if (ep + 1) % 10 == 0:
            # 1. Save the PyTorch model weights
            torch.save(agent.net.state_dict(), "mario_cnn_weights.pth")
            print("--> Model weights saved to mario_cnn_weights.pth")
            
            # 2. Update the progress graph
            save_progress_plot(episode_rewards)

if __name__ == "__main__":
    main()