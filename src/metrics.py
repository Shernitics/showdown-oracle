"""
Per-battle metrics collected during training.

BattleStats runs in each worker process and drops a row into info["battle"]
on the battle's final step. BattleLogger runs in the main process and writes
those rows out. They never import each other, so FIELDS is the contract.
"""
import csv
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from env import BROUGHT_SIZE

FIELDS = [
    "run", "steps", "opponent", "won", "turns", "reward", "tera",
    "mons", "mons_opp", "hp", "hp_opp", "r_switch", "r_move", "r_tera",
    "r_fix_pass", "r_fix_switch", "r_fix_move", "r_fix_tera",
    "r_friendly", "r_effective", "r_super", "r_no_effect", "tera_turn",
]


class BattleStats(gym.Wrapper):
    """
    Snapshots the battle on its last step into info["battle"].
    Goes outside DoubleAgentWrapper (needs the repaired action) and inside Monitor.
    """

    def __init__(self, env, vgc_env, opponent):
        super().__init__(env)
        self.vgc_env = vgc_env
        self.opponent = opponent
        self._restart()

    def _restart(self):
        self.ep_steps = 0
        self.tiers = [0, 0, 0, 0]                   # pass, switch, move, tera
        self.n_moves = 0                            # move/tera actions decoded
        self.n_attacks = 0                          # of those, damaging and aimed at a foe
        self.n_friendly = 0
        self.n_effective = 0
        self.n_super = 0
        self.n_no_effect = 0
        self.tera_turn = None
        self.repairs_before = list(self.env.repairs)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._restart()
        return out

    def step(self, action):

        battle = self.vgc_env.battle1
        counted = battle is not None and not battle.teampreview   # preview ints are team slots
        active = battle.active_pokemon if counted else None       # capture before the step:
        foes = battle.opponent_active_pokemon if counted else None  # battle1 advances below
        turn = battle.turn if counted else None

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ep_steps += 1

        if counted:
            for i, a in enumerate(self.env.last_action):   # post-repair, what the battle saw
                tier = self.env.gimmick_tier(int(a))
                self.tiers[tier] += 1

                if tier == 3 and self.tera_turn is None:
                    self.tera_turn = turn

                if tier in (2, 3):
                    self._decode(int(a), i, active, foes)

        if terminated or truncated:
            info["battle"] = self._snapshot(self.vgc_env.battle1)

        return obs, reward, terminated, truncated, info

    def _decode(self, action, i, active, foes):
        """
        Pulls move and target out of a move/tera action, mirroring poke-env's encoding.
        """

        mon = active[i] if active and i < len(active) else None
        if mon is None:
            return

        moves = list(mon.moves.values())[:4]
        idx = (action - 7) % 20 // 5
        if idx >= len(moves):               # poke-env falls back to available_moves here
            return

        move = moves[idx]
        target = (action - 7) % 5 - 2       # -2, -1 own slots, 0 none, 1, 2 foes
        self.n_moves += 1

        if target == -(2 - i) and move.base_power > 0:
            self.n_friendly += 1

        if target not in (1, 2) or move.base_power == 0:
            return

        foe = foes[target - 1] if foes and target - 1 < len(foes) else None
        if foe is None:
            return

        multiplier = foe.damage_multiplier(move)    # reads tera type when terastallized
        self.n_attacks += 1
        self.n_effective += multiplier >= 1
        self.n_super += multiplier >= 2
        self.n_no_effect += multiplier == 0

    def _snapshot(self, battle):

        fixes = [a - b for a, b in zip(self.env.repairs, self.repairs_before)]
        steps = max(self.ep_steps, 1)
        mine = [m for m in battle.team.values() if m.selected_in_teampreview]
        seen = list(battle.opponent_team.values())
        unseen = BROUGHT_SIZE - len(seen)
        total = max(sum(self.tiers), 1)
        moves = max(self.n_moves, 1)
        attacks = max(self.n_attacks, 1)

        return {
            "opponent": self.opponent,
            "won": battle.won,
            "turns": battle.turn,
            "r_fix_pass": fixes[0] / steps,
            "r_fix_switch": fixes[1] / steps,
            "r_fix_move": fixes[2] / steps,
            "r_fix_tera": fixes[3] / steps,
            "tera": sum(m.is_terastallized for m in mine),
            "mons": sum(not m.fainted for m in mine),
            "mons_opp": BROUGHT_SIZE - sum(m.fainted for m in seen),
            "hp": sum(m.current_hp_fraction for m in mine) / max(len(mine), 1),
            "hp_opp": (sum(m.current_hp_fraction for m in seen) + unseen) / BROUGHT_SIZE,
            "r_switch": self.tiers[1] / total,
            "r_move": self.tiers[2] / total,
            "r_tera": self.tiers[3] / total,
            "r_friendly": self.n_friendly / moves,
            "r_effective": self.n_effective / attacks,
            "r_super": self.n_super / attacks,
            "r_no_effect": self.n_no_effect / attacks,
            "tera_turn": self.tera_turn,
        }


class BattleLogger(BaseCallback):
    """
    Collects info["battle"] from every env, one CSV row per battle.
    """

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.n_runs = 0

    def _on_step(self):

        for info in self.locals["infos"]:
            row = info.get("battle")
            if row is None:
                continue

            self.n_runs += 1
            row["run"] = self.n_runs
            row["steps"] = self.num_timesteps
            row["reward"] = info.get("episode", {}).get("r")

            self._write(row)
            self.logger.record_mean("battle/won", float(bool(row["won"])))
            self.logger.record_mean("battle/turns", row["turns"])

        return True

    def _on_training_start(self):

        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as f:
                self.n_runs = sum(1 for _ in csv.DictReader(f))   # keep numbering across runs
            return

        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def _write(self, row):

        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
