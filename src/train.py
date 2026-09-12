import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from encode.vocab import FEATURE_VERSION
from env import VGCEnv
from extractor import VGCExtractor
from evaluate.results import add_run, connect
from teams import FORMAT, TEAM, TEAM_ID
from wrapper import MaskedSingleAgent

TOTAL_STEPS = 50_000
CHECKPOINT_FREQ = 25_000
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "vgc"


class Checkpoint(BaseCallback):

    def __init__(self, save_freq: int, save_dir: Path, run_id: str):
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.run_id = run_id
        self.episodes = 0
        self.start = time.perf_counter()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.episodes += int(np.sum(dones))
        if self.n_calls % self.save_freq == 0:
            self.save()
        return True

    def save(self):
        stem = self.save_dir / f"{self.run_id}_{self.num_timesteps}"
        self.model.save(stem)
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "train_steps": self.num_timesteps,
                    "train_episodes": self.episodes,
                    "wall_clock_s": round(time.perf_counter() - self.start, 1),
                }
            )
        )


def commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_env(seed: int | None = None):
    env = VGCEnv(
        battle_format=FORMAT,
        team=TEAM,
        strict=False,
        choose_on_teampreview=False,
    )
    opponent = RandomPlayer(
        battle_format=FORMAT, team=TEAM, start_listening=False
    )
    wrapped = MaskedSingleAgent(SingleAgentWrapper(env, opponent))
    if seed is not None:
        wrapped.reset(seed=seed)
    return wrapped


def main(seed: int = 0, total_steps: int = TOTAL_STEPS, checkpoint_freq: int = CHECKPOINT_FREQ):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:6]
    env = make_env(seed)

    hyperparams = {"n_steps": 2048, "batch_size": 64, "learning_rate": 3e-4}
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={"features_extractor_class": VGCExtractor},
        seed=seed,
        verbose=1,
        **hyperparams,
    )

    checkpoint = Checkpoint(checkpoint_freq, MODEL_DIR, run_id)
    start = time.perf_counter()
    model.learn(total_timesteps=total_steps, callback=checkpoint)
    wall_clock = round(time.perf_counter() - start, 1)
    checkpoint.save()
    model.save(MODEL_PATH)

    con = connect()
    add_run(
        con,
        (
            run_id,
            seed,
            commit_sha(),
            FEATURE_VERSION,
            FORMAT,
            TEAM_ID,
            json.dumps(hyperparams),
            str(model.device),
            wall_clock,
        ),
    )
    con.close()

    print(f"run {run_id}  seed {seed}  steps {env.n_steps}  repairs {env.n_repairs}  rate {env.repair_rate:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    args = parser.parse_args()
    main(args.seed, args.steps, args.checkpoint_freq)
