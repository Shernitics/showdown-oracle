import gymnasium as gym
import numpy as np
from poke_env.environment import DoublesEnv


FIRST_MOVE_ACTION = 7
TIER_SIZE = 20


def gimmick_tier(action: int) -> int:
    """
    Gimmicks
    0: pass
    1-6: switch to bench slot
    7-26: move (4 moves x 5 targets)
    27-46: move + mega
    47-66: move + Z-move
    67-86: move + Dynamax
    87-106: move + tera
    """
    if action < FIRST_MOVE_ACTION:
        return 0
    return (action - FIRST_MOVE_ACTION) // TIER_SIZE


class MaskedSingleAgent(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.half = DoublesEnv.get_action_space_size(9)
        self.observation_space = env.observation_space["observation"]
        self._mask = None
        self.n_repairs = 0
        self.n_steps = 0

    def _flatten(self, obs):
        self._mask = obs["action_mask"]
        return obs["observation"]

    def action_masks(self):
        return self._mask

    def _repair(self, action):
        a0, a1 = int(action[0]), int(action[1])

        tier = gimmick_tier(a0)
        if tier > 0 and gimmick_tier(a1) == tier:
            self.n_repairs += 1
            return np.array([a0, a1 - TIER_SIZE * tier], dtype=np.int64)

        if a0 == a1 and a0 < FIRST_MOVE_ACTION:
            legal = [i for i in np.flatnonzero(self._mask[self.half:]) if i != a0]
            if legal:
                self.n_repairs += 1
                return np.array([a0, int(np.random.choice(legal))], dtype=np.int64)

        return action

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
