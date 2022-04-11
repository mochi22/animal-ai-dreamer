import numpy as np
import gym
from gym.spaces.box import Box


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class DummyWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(DummyWrapper, self).__init__(env)

    def observation(self, observation):
        return observation


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        """PyTorchのミニバッチのインデックス順に変更するラッパー"""
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
