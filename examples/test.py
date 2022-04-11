# from stable_baselines3 import PPO
from stable_baselines3 import DQN
import torch as th

import sys
import random
import os

# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment

from dreamer.wrapper import WrapPyTorch, OneHotAction


def train_agent_single_config(configuration_file):

    aai_env = AnimalAIEnvironment(
        seed=123,
        file_name="../env/AnimalAI",
        arenas_configurations=configuration_file,
        play=False,
        base_port=5001,
        inference=False,
        useCamera=True,
        resolution=64,
        useRayCasts=False,
        # raysPerSide=1,
        # rayMaxDegrees = 30,
    )

    # env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=False)
    # def make_env():
    #     def _thunk():
    #         env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=True)
    #         return env
    #     return _thunk
    # env = DummyVecEnv([make_env()])
    env = UnityToGymWrapper(
        aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True
    )
    env = OneHotAction(WrapPyTorch(env))

    print(env.action_space.shape)
    print(env.observation_space)


# Loads a random competition configuration unless a link to a config is given as an argument.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]
    else:
        competition_folder = "../configs/competition/"
        configuration_files = os.listdir(competition_folder)
        configuration_random = random.randint(0, len(configuration_files))
        configuration_file = (
            competition_folder + configuration_files[configuration_random]
        )
    train_agent_single_config(configuration_file=configuration_file)
