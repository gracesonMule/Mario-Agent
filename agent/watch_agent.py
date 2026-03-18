import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import RecordVideo
import argparse
import os

# Import your custom agent and wrappers from your training script
from first_mario_agent import MarioAgent, SkipFrame, GrayScaleResizeWrapper, FrameStackWrapper

def watch_mario(inputdir):
    # 1. Initialize the environment identically to training
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)

    file_path = os.path.join(inputdir, 'mario_replays')

    env = RecordVideo(
        env, 
        video_folder=file_path,
        episode_trigger=lambda episode_id: True, # This tells it to record EVERY episode
        name_prefix='mario-agent'
    )

    env = SkipFrame(env, skip=4)
    env = GrayScaleResizeWrapper(env, shape=(84, 84))
    env = FrameStackWrapper(env, num_frames=4)

    # 2. Instantiate the agent AND load your saved weights
    # Make sure "mario_cnn_weights.pth" matches the filename you saved earlier!
    agent = MarioAgent(action_space_size=env.action_space.n, model_path=f"{inputdir}/mario_cnn_weights.pth")
    
    # 3. Shut down Epsilon-Greedy exploration
    # We want 100% exploitation of the trained neural network
    agent.exploration_rate = 0.0
    agent.exploration_rate_min = 0.0

    # 4. Start the visual loop
    episodes = 500
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"Watching Episode {ep + 1}...")
        
        while not done:
            # The agent evaluates the frame stack and picks an action
            action = agent.act(state)
            
            # The game advances
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render the game window to your screen
            env.render()
            
        print(f"Episode {ep + 1} finished with a score of: {total_reward}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model weights from path")
    parser.add_argument('--inputdir', help="Path to load model weights from.", default=None, required=True)
    args = parser.parse_args()
    watch_mario(inputdir=args.inputdir)