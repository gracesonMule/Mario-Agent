import gym_super_mario_bros
from nes_py.app.play_human import play_human

# Initialize the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# Launch the interactive window
play_human(env)