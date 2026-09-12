import gymnasium as gym
import numpy as np
from gymnasium import spaces
from poke_env.environment import DoublesEnv


SWITCH_ACTIONS = frozenset(range(1, 7))
MEGA_ACTIONS = frozenset(range(27, 47))

class MaskedSingleAgent(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.half = DoublesEnv.get_action_space_size(9)
        inner = env.observation_space["observation"]
        self.observation_space = spaces.Dict(
            {**{k: inner[k] for k in inner.spaces}, "action_mask": env.observation_space["action_mask"]}
        )
        self._mask = None
        self.n_repairs = 0
        self.n_steps = 0

    def _flatten(self, obs):
        self._mask = obs["action_mask"]
        return {**obs["observation"], "action_mask": obs["action_mask"]}

    def action_masks(self):
        return self._mask

    def _repair(self, action):
        a0, a1 = int(action[0]), int(action[1])
        banned = set()
        if a0 == a1 and a0 in SWITCH_ACTIONS:
            banned.add(a0)
        if a0 in MEGA_ACTIONS and a1 in MEGA_ACTIONS:
            banned |= MEGA_ACTIONS
        if not banned:
            return action

        legal = [i for i in np.flatnonzero(self._mask[self.half:]) if i not in banned]
        if not legal:
            return action

        self.n_repairs += 1
        return np.array([a0, int(np.random.choice(legal))], dtype=np.int64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        self.n_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(self._repair(action))
        return self._flatten(obs), reward, terminated, truncated, info

    @property
    def repair_rate(self) -> float:
        return self.n_repairs / max(self.n_steps, 1)
