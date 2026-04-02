import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from ultralytics import YOLO
from pynput import keyboard

# --- 1. KEYBOARD HANDLER ---
# Tracks whether a key is currently being held down
key_state = {'left': False, 'right': False, 'down': False, 'A': False, 'B': False}

def on_press(key):
    try:
        if key.char == 'a': key_state['left'] = True
        if key.char == 'd': key_state['right'] = True
        if key.char == 's': key_state['down'] = True
    except AttributeError:
        # Handle special keys (arrows, space, shift)
        if key == keyboard.Key.left: key_state['left'] = True
        if key == keyboard.Key.right: key_state['right'] = True
        if key == keyboard.Key.down: key_state['down'] = True
        if key == keyboard.Key.space: key_state['A'] = True
        if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]: 
            key_state['B'] = True

def on_release(key):
    try:
        if key.char == 'a': key_state['left'] = False
        if key.char == 'd': key_state['right'] = False
        if key.char == 's': key_state['down'] = False
    except AttributeError:
        if key == keyboard.Key.left: key_state['left'] = False
        if key == keyboard.Key.right: key_state['right'] = False
        if key == keyboard.Key.down: key_state['down'] = False
        if key == keyboard.Key.space: key_state['A'] = False
        if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]: 
            key_state['B'] = False

# Start listening to the keyboard in the background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def get_mario_action():
    """Maps the current physical keyboard state to the correct Gym Action Integer."""
    if key_state['right']:
        if key_state['A'] and key_state['B']: return 4 # Right + Jump + Run
        if key_state['A']: return 2                    # Right + Jump
        if key_state['B']: return 3                    # Right + Run
        return 1                                       # Right
    elif key_state['left']:
        if key_state['A'] and key_state['B']: return 9 # Left + Jump + Run
        if key_state['A']: return 7                    # Left + Jump
        if key_state['B']: return 8                    # Left + Run
        return 6                                       # Left
    elif key_state['down']:
        return 10                                      # Duck
    elif key_state['A']:
        return 5                                       # Jump in place
    return 0                                           # NOOP (Do nothing)


# --- 2. YOLO AND ENV SETUP ---
model = YOLO('runs/detect/train6/weights/best.pt')

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT) # Upgraded to allow backwards jumping

state = env.reset()
done = False

print("--- CONTROLS ---")
print("Move: A/D or Left/Right Arrows")
print("Jump: Spacebar")
print("Run/Fire: Shift")
print("Quit: Press 'q' in the video window")
print("----------------")

# --- 3. THE GAME LOOP ---
while not done:
    # Get human input
    action = get_mario_action()
    
    # Step environment
    state, reward, done, info = env.step(action)
    
    # Fix colors
    full_frame_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    masked_input = full_frame_bgr.copy()
    
    # Apply UI Masking for YOLO
    masked_input[0:31, :] = (0, 0, 0)   
    masked_input[224:240, :] = (0, 0, 0) 
    
    # Run YOLO on masked input
    results = model(masked_input, verbose=False)
    
    # Draw boxes on the unmasked frame
    annotated_frame = results[0].plot(img=full_frame_bgr)
    
    # Scale up for visibility
    display_frame = cv2.resize(annotated_frame, (768, 720), interpolation=cv2.INTER_NEAREST)
    
    # Display the game
    cv2.imshow("Super Mario Bros - YOLO Real-Time", display_frame)
    
    # WaitKey is still required to actually render the OpenCV window and catch the quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if done:
        state = env.reset()
        done = False 

env.close()
cv2.destroyAllWindows()