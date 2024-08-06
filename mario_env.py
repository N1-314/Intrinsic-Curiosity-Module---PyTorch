import gym
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation, LazyFrames, RecordVideo
from gym.spaces import Box
# https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT)
from nes_py.wrappers import JoypadSpace
from collections import deque
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from utils import create_video
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_mario_env(env_id = "SuperMarioBros-1-1-v0", color_env = False, save_video = True, test=False):
    if test:
        num_skip = 1
    else:
        num_skip = 6
    gray_scale = not color_env
    ## Initialize Super Mario environment
    if gym.__version__ < '0.26':
        env = gym.make(env_id, new_step_api=True)
    else:
        env = gym.make(env_id, apply_api_compatibility=True)
    # wrap the environment
    env = JoypadSpace(env, COMPLEX_MOVEMENT) # "we reparametrize the action space of the agent into 14 unique actions"
    env = RecordFrames(env, COMPLEX_MOVEMENT) if save_video else env
    env = ResizeObservation(env, shape=42) # ndim을 3으로 만들기에 GrayScale이전에 사용
    env = GrayScaleObservation(env, keep_dim=False) if gray_scale else env # "The input RGB images are converted into gray-scale"
    env = SkipStackObservation(env, num_skip=num_skip, num_stack=4)

    return env

class MarioJoystick():
    def __init__(self, actions):
        self.actions = actions

        x, y = 50, 50 # center
        l, s, t = 25, 20, 10 # long, short, triangle
        offset = 2
        # draw.circle((x, y),offset, outline=(0,0,0,255), width=2)
        self.dir_button_positions = {
            'up': [(x, y -offset), (x-s//2, y-t -offset), (x-s//2, y-(l+t) -offset), (x+s//2, y-(l+t) -offset), (x+s//2, y-t -offset)], # up
            'down': [(x, y +offset), (x-s//2, y+t +offset), (x-s//2, y+(l+t) +offset), (x+s//2, y+(l+t) +offset), (x+s//2, y+t +offset)], # down
            'left': [(x -offset, y), (x-t -offset, y-s//2), (x-(l+t) -offset, y-s//2), (x-(l+t) -offset, y+s//2), (x-t -offset, y+s//2)], # left
            'right': [(x +offset, y), (x+t +offset, y-s//2), (x+t+l +offset, y-s//2), (x+t+l +offset, y+s//2), (x+t +offset, y+s//2)], # right
        }
        self.r = s/2
        self.ab_button_positions = { # A is jump, B is run(power-up)
            'A': (x+60, y-20), # A
            'B': (x+60, y+20) # B
        }

    def draw(self, action):
        press = self.actions[action]
         
        joystick_image = Image.new('RGBA', (135, 100), color=(255, 255, 255, 50))
        # joystick_image = Image.new('RGB', (135, 100), color=(10, 10, 10))
        # joystick_image = Image.fromarray(img)
        
        draw = ImageDraw.Draw(joystick_image)
        for key, pos in self.dir_button_positions.items():
            if key in press:
                draw.polygon(pos, fill='blue')
            draw.polygon(pos, outline=(0,0,0,255), width=2)
        for key, pos in self.ab_button_positions.items():
            if key in press:
                draw.circle(pos, self.r, outline=(0,0,0,255), fill='blue')
            draw.circle(pos, self.r, outline=(0,0,0,255), width=2)
            if key in press:
                draw.text((pos[0]-3, pos[1]-5), key, font=ImageFont.load_default())
            else:
                draw.text((pos[0]-3, pos[1]-5), key, font=ImageFont.load_default(), fill=(0, 0, 0, 255))
        
        return joystick_image
    
    def __call__(self, obs, action):
        overlay = np.array(self.draw(action))
        overlay = cv2.resize(overlay, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        height, width, c = overlay.shape
        
        x_offset, y_offset = 0, 100
        y1, y2 = y_offset, y_offset + height
        x1, x2 = x_offset, x_offset + width

        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(3):
            obs[y1:y2, x1:x2, c] = (alpha_overlay * overlay[:, :, c] + alpha_background * obs[y1:y2, x1:x2, c])

        return obs.copy()

class RecordFrames(gym.Wrapper):
    def __init__(self, env: gym.Env, actions=None):
        super().__init__(env)
        self.record_frames = []
        self.show_joystick = True if actions is not None else False
        if self.show_joystick:
            self.joystick = MarioJoystick(actions)
        self.record = True
        self.total_reward = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        if self.record and self.show_joystick:
            if self.show_joystick:
                observation = self.joystick(observation, action)
                cv2.putText(observation, f"x_pos: {info['x_pos']}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(observation, f"reward: {reward}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(observation, f"total_reward: {self.total_reward}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.record_frames.append(observation.copy())
            info['n_frames'] = len(self.record_frames)

        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.record_frames.clear()
        self.record_frames.append(observation.copy())
        self.total_reward = 0

        return observation, info
    
    def save_video(self, name='output.mp4', fps=20):
        create_video(self.record_frames, name, fps=fps, is_color=True)
        print(f'[save video] saved: {name}, frames length: {len(self.record_frames)}, fps: {fps}')

class SkipStackObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, num_skip=6, num_stack=4):
        super().__init__(env)
        self.num_skip = num_skip
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)

        if num_stack!=1:
            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
            self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def observation(self, observation):
        if self.num_stack!=1:
            return np.array(self.frames)
        else:
            return self.frames[0]
        
    def step(self, action):
        total_reward = 0
        for _ in range(self.num_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        self.frames.append(observation)
            
        return self.observation(None), total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        for _ in range(self.num_stack):
            self.frames.append(observation)

        return self.observation(None), info

if __name__ == '__main__':
    env = create_mario_env()
    
    for episode in range(1):
        obs, info = env.reset()
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated or info['flag_get']:
                break
    
    env.save_video('output.mp4', fps=30)
    env.close()