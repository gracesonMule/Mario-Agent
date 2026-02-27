import gym
import numpy as np
import cv2
from gym.spaces import Box

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class GrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        
        # Update the environment's observation space to reflect our new dimensions
        # It will now expect an 84x84 image with 1 color channel (grayscale)
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

class MarioAgent:
    def __init__(self, action_space_size):
        # Initialize your neural network here
        self.action_space_size = action_space_size
        print(f"Agent initialized with {action_space_size} possible actions.")

    def act(self, observation):
        """
        This is where your neural network makes a decision.
        
        Input: 'observation' is a numpy array of the game screen (240x256x3 RGB pixels).
        Output: An integer representing the chosen action.
        """
        
        # TODO: Pass the observation through your neural network
        # Example: action = self.my_neural_network.forward(observation)
        
        # For now, we will just return a random action to test the loop
        import random
        action = random.randint(0, self.action_space_size - 1)
        
        return action

def main():
    # 1. Initialize the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    # 2. Restrict the action space to standard Mario movements (0 to 6)
    # This makes it much easier for a neural network to learn
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    env = GrayScaleResizeWrapper(env, shape=(84, 84))

    # 3. Instantiate your agent
    # SIMPLE_MOVEMENT has 7 discrete actions, so action_space.n will be 7
    agent = MarioAgent(action_space_size=env.action_space.n)
    
    # 4. Start the game loop, how times to run
    episodes = 3

    # save inputs of previous runs
    # 
    
    for ep in range(episodes):
        # Reset the environment for a new game
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"Starting Episode {ep + 1}")
        
        while not done:
            # The agent looks at the screen and picks an action
            action = agent.act(state)
            
            # The environment takes that action and returns the next frame and reward
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render the game window so you can watch your agent play
            env.render()
            
        print(f"Episode {ep + 1} finished with a total reward of: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()