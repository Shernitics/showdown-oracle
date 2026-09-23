"""
Adapter between poke-env's SingleAgentWrapper and MaskablePPO.

Two major roles:
- Splits observation and action mask, the observation is returned normally whereas the mask is held here.
- Repairs when a pair of moves, which are individually legal but jointly illegal such as terastallizing together, switching to the same Pokémon or both pass.
"""
import gymnasium as gym
import numpy as np
from poke_env.environment import DoublesEnv

TERA_OFFSET = 80

class DoubleAgentWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.half = DoublesEnv.get_action_space_size(9)                 # splits complete action mask into 2 (left and right actions)
        self.observation_space = env.observation_space["observation"]
        self._mask = None
        self.n_steps = 0
        self.n_repairs = 0
        self.repairs = [0, 0, 0, 0]         # 0 - pass, 1 - switch, 2 - move, 3 - tera
        self.last_action = None

    @staticmethod
    def gimmick_tier(action):
        """
        converts action (int) into gimmick tier.
        """

        # Generation 9 vanilla, hence only tera gimmick
        # 0 - pass, 1 - switch, 2 - move, 3 - tera

        action_to_gimmick = {
            0 : [0],
            1 : list(range(1, 6+1)),
            2 : list(range(7, 26+1)),
            3 : list(range(87, 106+1)),
        }

        for tier, a in action_to_gimmick.items():
            if action in a:
                return tier

        else:
            raise ValueError("Action out of range")

    def _repair(self, actions):

        a0, a1 = actions
        g0, g1 = self.gimmick_tier(a0), self.gimmick_tier(a1)

        if g0 == g1 == 3:                       # one tera per battle, so the right slot keeps it
            demoted = a0 - TERA_OFFSET          # same move and target, minus the gimmick
            if self._mask[demoted]:
                self.n_repairs += 1
                self.repairs[3] += 1
                return np.array([demoted, a1], dtype=np.int64)

        if a0 == a1 and g0 in (0, 1):           # both slots want the same switch, or both pass
            left, right = np.flatnonzero(self._mask[:self.half]),  np.flatnonzero(self._mask[self.half:])
            left, right = left[left != a0], right[right != a1]

            if left.size:                       # lowest legal alternative, never a random one
                repaired = [int(left[0]), a1]
            elif right.size:
                repaired = [a0, int(right[0])]
            else:
                return actions                  # nothing to swap to; poke-env handles it

            self.n_repairs += 1
            self.repairs[g0] += 1
            return np.array(repaired, dtype=np.int64)

        return actions

    def _split(self, observation):
        self._mask = observation["action_mask"]
        return observation["observation"]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._split(observation), info

    def step(self, action):
        self.n_steps += 1
        self.last_action = self._repair(action)
        observation, reward, terminated, truncated, info = self.env.step(self.last_action)
        return self._split(observation), reward, terminated, truncated, info

    @property
    def repair_rate(self):
        return self.n_repairs / max(self.n_steps, 1)

    def action_masks(self):
        return self._mask
