import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from ultralytics import YOLO

# 1. Load your trained model
# Point this to the best.pt file from your most recent training run
model = YOLO('runs/detect/train5/weights/best.pt')

# 2. Initialize the Mario Environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT) # Limits actions to standard D-pad + jump

state = env.reset()
done = False

print("Starting Mario... Press 'q' in the video window to quit.")

while not done:
    # 3. Choose an action
    # Currently taking random actions. You can easily plug this right into 
    # the decision loop of your RL agent instead of sampling the action space.
    action = env.action_space.sample()
    
    # 4. Step the environment forward (Gym 0.21.0 format)
    state, reward, done, info = env.step(action)
    
    # 5. Fix the color space (RGB to BGR)
    frame_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    
    # 6. Run YOLO inference directly on the numpy array
    # verbose=False stops YOLO from printing to the terminal every single frame
    results = model(frame_bgr, verbose=False)
    
    # 7. Draw the bounding boxes
    # results[0].plot() automatically draws the boxes, labels, and confidences on the frame
    annotated_frame = results[0].plot()
    
    # 8. Scale up the video window (Native NES resolution is tiny: 256x240)
    display_frame = cv2.resize(annotated_frame, (768, 720), interpolation=cv2.INTER_NEAREST)
    
    # 9. Show the result
    cv2.imshow("Super Mario Bros - YOLO Real-Time", display_frame)
    
    # 10. Check for the 'q' key to cleanly exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Reset the environment if Mario dies or finishes the level
    if done:
        state = env.reset()
        done = False # Keep the loop running for the next life

# Clean up windows when finished
env.close()
cv2.destroyAllWindows()