import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

# 1. Initialize the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 2. Preprocess the visual inputs for the architecture
env = GrayScaleObservation(env, keep_dim=True)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# 3. Define the Agent Architecture
# We use 'CnnPolicy' (Convolutional Neural Network) because the agent is learning directly from 2D image pixels
model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-4)

# 4. Train the Agent
print("Starting training...")
model.learn(total_timesteps=100000)

# 5. Save the "Brain"
model.save("mario_ppo_model")