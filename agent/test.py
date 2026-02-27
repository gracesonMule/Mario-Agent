import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Initialize the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# Limit the action-space to simple movements (run, jump, right, left)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Run a quick test loop
state = env.reset()
done = False

print("Starting Mario environment... press Ctrl+C in the terminal to quit.")

while not done:
    # Choose a completely random action
    action = env.action_space.sample() 
    state, reward, done, info = env.step(action)
    
    # Render the game window
    env.render()

env.close()