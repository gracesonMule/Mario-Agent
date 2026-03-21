import os
import cv2
import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# --- CONFIGURATION ---
# Change this to play different levels. Format: SuperMarioBros-<World>-<Stage>-v0
level_to_play = 'SuperMarioBros-1-1-v0' 
output_dir = "mario_dataset/human_play_images"
save_interval = 30 # Saves an image every 30 frames (0.5 seconds)

output_dir = os.path.join(output_dir, level_to_play)
os.makedirs(output_dir, exist_ok=False)


def get_action_from_keyboard():
    """Maps Pygame keyboard presses to Mario's COMPLEX_MOVEMENT index."""
    keys = pygame.key.get_pressed()
    action_list = []

    # D-Pad
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: action_list.append('right')
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action_list.append('left')
    if keys[pygame.K_UP] or keys[pygame.K_w]: action_list.append('up')
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action_list.append('down')
    
    # Buttons
    if keys[pygame.K_SPACE]: action_list.append('A') # Jump
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action_list.append('B') # Run/Fireball

    # Map the pressed keys to the closest action index in COMPLEX_MOVEMENT
    if 'right' in action_list:
        if 'A' in action_list and 'B' in action_list: return 4
        if 'B' in action_list: return 3
        if 'A' in action_list: return 2
        return 1
    if 'left' in action_list:
        if 'A' in action_list and 'B' in action_list: return 9
        if 'B' in action_list: return 8
        if 'A' in action_list: return 7
        return 6
    if 'A' in action_list: return 5
    if 'down' in action_list: return 10
    if 'up' in action_list: return 11
    
    return 0 # NOOP (Do nothing)

# --- INITIALIZATION ---
env = gym_super_mario_bros.make(level_to_play)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

pygame.init()
# Mario's native resolution is 256x240. We scale it x2 so you can actually see it.
screen = pygame.display.set_mode((512, 480))
pygame.display.set_caption(f"Playing {level_to_play} - Recording Data")
clock = pygame.time.Clock()

state = env.reset()
done = False
frame_count = 0
saved_count = 0
running = True

print(f"Starting {level_to_play}! Close the Pygame window to stop and save.")
print("Controls: WASD/Arrows to move, Space to Jump, Shift to Run/Fireball")

# --- GAME LOOP ---
while running:
    # 1. Keep the game running at ~60 FPS so it's playable
    clock.tick(60)
    
    # 2. Handle quitting the Pygame window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 3. Get keyboard input and step the environment
    action = get_action_from_keyboard()
    state, reward, done, info = env.step(action)

    # 4. Render the game frame to the Pygame window
    # Numpy array is (240, 256, 3). Pygame expects (256, 240, 3). We swap axes.
    surface = pygame.surfarray.make_surface(state.swapaxes(0, 1))
    scaled_surface = pygame.transform.scale(surface, (512, 480))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()

    # 5. Extract and save the frame
    if frame_count % save_interval == 0:
        frame_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        filename = os.path.join(output_dir, f"human_frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame_bgr)
        saved_count += 1
        
    frame_count += 1

    # 6. Infinite attempts logic: If you die or beat the level, it resets.
    if done:
        print("Level finished or Mario died! Resetting for another attempt...")
        state = env.reset()
        done = False

pygame.quit()
env.close()
print(f"Session ended. Successfully saved {saved_count} frames to '{output_dir}'.")