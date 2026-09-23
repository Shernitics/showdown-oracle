"""
Per-battle metrics collected during training.

BattleStats runs in each worker process and drops a row into info["battle"]
on the battle's final step. BattleLogger runs in the main process and writes
those rows out. They never import each other, so FIELDS is the contract.

BattleStats also guards against hung battles: an episode that stops advancing is
truncated instead of consuming its worker forever, and BattleLogger warns when an
opponent stops producing battles. Both exist because a stalled worker is otherwise
invisible - it writes no rows at all, so the only symptom is a win rate that drifts
for no reason.
"""
import csv
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from env import BROUGHT_SIZE

# A battle whose turn counter has not moved in this many steps is stuck. The
# longest legitimate battle seen over 69k logged battles was 108 turns (~230 env
# steps) and p99 was 21 turns, so this leaves a wide margin.
STALL_STEPS = 150
MAX_EPISODE_STEPS = 500

# battles.csv grows about 4.5MB an hour, so a run left going for days needs a
# ceiling. Rolled files keep the same battles-<stamp>.csv naming as the schema
# archive, and the size is only checked every so often to avoid a stat() per row.
ROTATE_BYTES = 128 * 1024 * 1024
ROTATE_CHECK_EVERY = 500


def _free_name(path):
    """
    A path that does not exist yet. Timestamps collide when the log rolls twice
    inside one second, and copying onto an existing archive destroys it silently.
    """

    if not path.exists():
        return path

    for n in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate

    return path


def _last_run(path):
    """
    Highest run number in a log, read from the tail rather than by walking the
    file. Rotation moves rows out, so counting them would restart the numbering
    every time the log rolls.
    """

    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - 8192))
            tail = f.read().splitlines()
    except OSError:
        return 0

    for line in reversed(tail):
        first = line.split(b",", 1)[0].strip()
        if first.isdigit():
            return int(first)

    return 0

FIELDS = [
    "run", "steps", "opponent", "won", "turns", "reward", "tera",
    "mons", "mons_opp", "hp", "hp_opp", "r_switch", "r_move", "r_tera",
    "r_fix_pass", "r_fix_switch", "r_fix_move", "r_fix_tera",
    "r_friendly", "r_effective", "r_super", "r_no_effect", "tera_turn",
    "capped",
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
        self.n_capped = 0
        self._restart()

    def _restart(self):
        self.ep_steps = 0
        self.last_turn = None
        self.turn_stuck_for = 0
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

        now = self.vgc_env.battle1
        capped = False

        if not (terminated or truncated) and now is not None:
            # a hang shows up as steps piling up while `turn` stands still
            if now.turn == self.last_turn:
                self.turn_stuck_for += 1
            else:
                self.last_turn = now.turn
                self.turn_stuck_for = 0

            if self.turn_stuck_for >= STALL_STEPS or self.ep_steps >= MAX_EPISODE_STEPS:
                self._report_hang(now)
                truncated = True
                capped = True
                self.n_capped += 1

        if terminated or truncated:
            info["battle"] = self._snapshot(self.vgc_env.battle1, capped)

        return obs, reward, terminated, truncated, info

    def _report_hang(self, battle):
        """
        Dump enough state to diagnose the hang before truncating. Guarded
        throughout: the diagnostic must never be what kills the worker.
        """

        def safe(fn, default="?"):
            try:
                return fn()
            except Exception:
                return default

        reason = "no turn progress" if self.turn_stuck_for >= STALL_STEPS else "step cap"

        print(
            f"\n[HANG] {datetime.now():%H:%M:%S} opponent={self.opponent} reason={reason}\n"
            f"       ep_steps={self.ep_steps} turn={safe(lambda: battle.turn)} "
            f"stuck_for={self.turn_stuck_for} teampreview={safe(lambda: battle.teampreview)}\n"
            f"       battle_tag={safe(lambda: battle.battle_tag)}\n"
            f"       last_action={safe(lambda: list(self.env.last_action))} "
            f"force_switch={safe(lambda: battle.force_switch)} "
            f"trapped={safe(lambda: list(battle.trapped))}\n"
            f"       active={safe(lambda: [getattr(m, 'species', None) for m in battle.active_pokemon])} "
            f"foes={safe(lambda: [getattr(m, 'species', None) for m in battle.opponent_active_pokemon])}\n"
            f"       avail_moves={safe(lambda: [[mv.id for mv in s] for s in battle.available_moves])}\n"
            f"       avail_switches={safe(lambda: [[p.species for p in s] for s in battle.available_switches])}\n",
            file=sys.stderr, flush=True,
        )

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

    def _snapshot(self, battle, capped=False):

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
            "capped": int(capped),
        }


class BattleLogger(BaseCallback):
    """
    Collects info["battle"] from every env, one CSV row per battle.

    Also watches opponent volume. A worker that quietly stops producing battles
    is otherwise invisible here, which silently reweights the opponent mix and
    makes the aggregate win rate move for no reason.
    """

    WINDOW = 600            # battles kept for the share check
    CHECK_EVERY = 200       # battles between checks

    def __init__(self, path, expected=()):
        super().__init__()
        self.path = Path(path)
        self.n_runs = 0
        self.n_capped = 0
        self.recent = deque(maxlen=self.WINDOW)
        # seeded rather than learned: an opponent discovered only from the rows it
        # produces cannot be missed when it produces none, which is exactly the
        # case worth catching. Each cycle starts a fresh logger, so a worker that
        # fails to come up at all would otherwise go unnoticed for that whole cycle.
        self.seen_opponents = set(expected)
        self.since_check = 0
        self.since_size_check = 0
        self.warned = set()

    def _on_step(self):

        for info in self.locals["infos"]:
            row = info.get("battle")
            if row is None:
                continue

            self.n_runs += 1
            row["run"] = self.n_runs
            row["steps"] = self.num_timesteps
            row["reward"] = info.get("episode", {}).get("r")

            if row.get("capped"):
                self.n_capped += 1
                print(f"[capped]    run {self.n_runs} vs {row['opponent']} truncated as hung "
                      f"({self.n_capped} so far)", flush=True)

            self._write(row)
            self.recent.append(row["opponent"])
            self.seen_opponents.add(row["opponent"])
            self.since_check += 1

            self.logger.record_mean("battle/won", float(bool(row["won"])))
            self.logger.record_mean("battle/turns", row["turns"])

        self._check_shares()
        return True

    def _check_shares(self):
        """
        Warn when an opponent falls below half its expected share of recent
        battles. Fires on partial degradation, not just total silence: a worker
        winding down from 160 battles per bucket to 20 is already broken.
        """

        if self.since_check < self.CHECK_EVERY or len(self.recent) < self.WINDOW:
            return

        self.since_check = 0
        expected = len(self.recent) / max(len(self.seen_opponents), 1)

        for opponent in sorted(self.seen_opponents):
            count = sum(1 for o in self.recent if o == opponent)
            degraded = count < expected / 2

            if degraded and opponent not in self.warned:
                print(f"[DEGRADED]  {opponent}: {count} of last {len(self.recent)} battles "
                      f"(expected ~{expected:.0f}) at {self.num_timesteps} steps", flush=True)
                self.warned.add(opponent)
            elif not degraded and opponent in self.warned:
                print(f"[recovered] {opponent}: {count} of last {len(self.recent)} battles "
                      f"at {self.num_timesteps} steps", flush=True)
                self.warned.discard(opponent)

            self.logger.record(f"battle/share_{opponent}", count / max(len(self.recent), 1))

    def _on_training_start(self):

        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), None)

            if header == FIELDS:
                self.n_runs = self._resume_run()    # keep numbering across runs and rotations
                return

            # Schema changed, so old rows cannot share this file. Copy them aside
            # rather than renaming: on Windows an editor or indexer holding the log
            # open still permits read and write but blocks rename, and bookkeeping
            # must never be what takes down a long run.
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            archived = _free_name(self.path.with_name(f"{self.path.stem}-{stamp}{self.path.suffix}"))
            try:
                shutil.copyfile(self.path, archived)
                print(f"[archived]  {archived.name} (CSV schema changed)", flush=True)
            except OSError as exc:
                print(f"[warning]   could not archive old log ({type(exc).__name__}); "
                      f"it will be overwritten", flush=True)

        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def _resume_run(self):
        """
        Where run numbering should pick up. Checks the newest rolled file too,
        since the live log is nearly empty just after a rotation.
        """

        archives = sorted(                      # by mtime: a "-2" suffix from a same-second
            self.path.parent.glob(f"{self.path.stem}-*{self.path.suffix}"),   # collision sorts
            key=lambda p: p.stat().st_mtime,                                  # before the plain
        )                                                                     # name, not after
        newest = _last_run(archives[-1]) if archives else 0
        return max(_last_run(self.path), newest)

    def _rotate(self):
        """
        Roll the log once it gets large. Copies then truncates rather than
        renaming: on Windows an editor indexing the file blocks a rename but not
        reads or writes, and a permanent run must not stop over a log move.
        """

        try:
            if self.path.stat().st_size < ROTATE_BYTES:
                return
        except OSError:
            return

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rolled = _free_name(self.path.with_name(f"{self.path.stem}-{stamp}{self.path.suffix}"))

        try:
            shutil.copyfile(self.path, rolled)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()
            print(f"[rotated]   {rolled.name} at {self.num_timesteps} steps", flush=True)
        except OSError as exc:
            print(f"[warning]   log rotation failed ({type(exc).__name__}); continuing",
                  flush=True)

    def _write(self, row):

        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        self.since_size_check += 1
        if self.since_size_check >= ROTATE_CHECK_EVERY:
            self.since_size_check = 0
            self._rotate()
