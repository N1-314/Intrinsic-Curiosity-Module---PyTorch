import warnings
from collections import deque

import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT, RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
from nes_py.wrappers import JoypadSpace

from utils import create_video

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_mario(env_id = "SuperMarioBros-1-1-v0", color_env = False, save_video = True, num_skip=6, num_stack=4):
    gray_scale = not color_env
    ## Initialize Super Mario environment
    if gym.__version__ < '0.26':
        env = gym.make(env_id, new_step_api=True)
    else:
        env = gym.make(env_id, apply_api_compatibility=True) # render_mode='human'
    
    if save_video:
        env = RecordFrames(env)
    
    env = JoypadSpace(env, RIGHT_ONLY)    # "we reparametrize the action space of the agent into 12 unique actions"
    env = ResizeObservation(env, shape=42) # ndim을 3으로 만들기에 GrayScale이전에 사용
    if gray_scale:
        env = GrayScaleObservation(env, keep_dim=False)     # "The input RGB images are converted into gray-scale"
    env = Div255Observation(env)
    env = SkipStackObservation(env, num_skip=num_skip, num_stack=num_stack)

    return env

class RecordFrames(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.record_frames = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.record_frames.append(observation.copy())
        info['n_frames'] = len(self.record_frames)

        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.record_frames.clear()
        self.record_frames.append(observation.copy())

        return observation, info
    
    def save_video(self, name='output.mp4', fps=20):
        create_video(self.record_frames, name, fps=fps, is_color=True)
        print(f'[save video] saved: {name}, frames length: {len(self.record_frames)}, fps: {fps}')


class Div255Observation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def observation(self, observation: np.ndarray):
        return observation/255


class SkipStackObservation(gym.ObservationWrapper): # env_wrapper.BufferedObsEnv
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
            return np.array(self.frames).astype(np.float32)
        else:
            return self.frames[0].astype(np.float32)    # << 확인 필요
        
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
    env = create_mario(color_env=True)
    
    for episode in range(1):
        obs, info = env.reset()
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated or info['flag_get']:
                break
    
    env.save_video(fps=30)
    env.close()
