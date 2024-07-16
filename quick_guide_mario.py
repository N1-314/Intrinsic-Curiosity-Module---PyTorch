
import warnings

import gym
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT, RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
from nes_py.wrappers import JoypadSpace

from utils import array_to_image, create_video

warnings.filterwarnings("ignore", category=UserWarning)


## Variables
color_env = True
save_video = True


## Initialize Super Mario environment
if gym.__version__ < '0.26':
    env = gym.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym.make("SuperMarioBros-1-1-v0", 
                #    render_mode='human', 
                   apply_api_compatibility=True,
                   )
env = JoypadSpace(env, COMPLEX_MOVEMENT)    # "we reparametrize the action space of the agent into 14 unique actions"
if not color_env:
    env = GrayScaleObservation(env, keep_dim=False)     # "The input RGB images are converted into gray-scale"
env = ResizeObservation(env, shape=42)
env = FrameStack(env, num_stack=4)  # "by concatenating the current frame with the three previous frames"

"""
TODO - wrappers
 : SkipFrame() 'we use action repeat of four during training time in VizDoom and action repeat of six in Mario.'
"""

## Check the output from environment
state, info = env.reset()
next_state, reward, termin, trunc, info = env.step(action=0)
print(f"""
      NEXT_STATE {next_state.shape},
      REWARD {reward},
      TERMIN {termin},
      TRUNC {trunc},
      INFO {info}\n""")


## How to get a screenshot
if isinstance(state, np.ndarray):
    img = array_to_image(state)
    img.save('output.png')
    print("An image has been created: output.png", flush=True)
elif isinstance(state, gym.wrappers.frame_stack.LazyFrames):
    img = array_to_image(state[-1])
    img.save('output.png')
    print("An image has been created: output.png", flush=True)


## How to save video - 1
if save_video:
    frames = []

def collect_frames(state):
    if isinstance(state, np.ndarray):
        frames.append(np.array(state))
    elif isinstance(state, gym.wrappers.frame_stack.LazyFrames):
        frames.append(np.array(state[-1]))
    return


## See how the game works
num_epi = 1
for epi in range(num_epi):

    ret = 0
    state, info = env.reset()
    termin, trunc = False, False

    if save_video:
        collect_frames(state)

    for step in range(200):
        action = env.action_space.sample()
        next_state, reward, termin, trunc, info = env.step(action)
        ret += reward
        state = next_state

        if save_video:
            collect_frames(state)

        if termin or trunc or info['flag_get']:
            ret = 0
            observation, info = env.reset()
            termin, trunc = False, False

            break
    
env.close()


# How to save video - 2
if save_video:
    create_video(frames, 'output.mp4', fps=20, is_color=color_env)
    
