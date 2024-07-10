
# https://github.com/openai/gym/blob/master/gym/wrappers/gray_scale_observation.py
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/transform_observation.py


import gym
import numpy as np


class GrayScaleObservation(gym.wrappers.GrayScaleObservation):
    """ Convert the image observation from RGB to gray scale.

    Example:
        >>> env = gym.make("SuperMarioBros-v0")
        >>> env.observation_space
        Box(0, 255, (240, 256, 3), uint8)
        >>> env = GrayScaleObservation(gym.make("SuperMarioBros-v0"))
        Box(0, 255, (240, 256), uint8)
        >>> env = GrayScaleObservation(gym.make("SuperMarioBros-v0"), keep_dim=True)
        Box(0, 255, (240, 256, 1), uint8)
    """

    def __init__(self, env: gym.Env, keep_dim: bool=False):
        super().__init__(env, keep_dim)

    def observation(self, observation):
        observation = np.sum(
            np.multiply(observation, np.array([0.2125, 0.7154, 0.0721])), axis=-1
        ).astype(np.uint8)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        
        return observation
