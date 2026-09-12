import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env.environment import SingleAgentWrapper
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env import VGCEnv
from evaluate.results import add_battles, add_run, connect, done_cells
from teams import FORMAT, TEAM
from wrapper import MaskedSingleAgent

BROUGHT_SIZE = 4
MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
OPPONENTS = {
    "RandomPlayer": RandomPlayer,
    "MaxBasePowerPlayer": MaxBasePowerPlayer,
    "SimpleHeuristicsPlayer": SimpleHeuristicsPlayer,
}


def make_eval_env(opponent_name: str):
    env = VGCEnv(
        battle_format=FORMAT,
        team=TEAM,
        strict=False,
        choose_on_teampreview=False,
    )
    opponent = OPPONENTS[opponent_name](
        battle_format=FORMAT, team=TEAM, start_listening=False
    )
    return MaskedSingleAgent(SingleAgentWrapper(env, opponent))


def checkpoints(model_dir: Path) -> list:
    found = []
    for meta_path in sorted(model_dir.glob("*.json")):
        model_path = meta_path.with_suffix(".zip")
        if model_path.exists():
            found.append((model_path, json.loads(meta_path.read_text())))
    return sorted(found, key=lambda c: c[1]["train_steps"])


def play(model, env, meta: dict, opponent: str, deterministic: bool, eval_seed: int, n_battles: int) -> list:
    rows = []
    obs, _ = env.reset(seed=eval_seed)
    for battle_idx in range(n_battles):
        done, reward, truncated = False, 0.0, False
        while not done:
            action, _ = model.predict(
                obs, action_masks=get_action_masks(env), deterministic=deterministic
            )
            obs, step_reward, terminated, truncated, _ = env.step(action)
            reward += step_reward
            done = terminated or truncated

        battle = env.env.env.battle1
        if truncated and not battle.finished:
            end_reason = "timeout"
        elif battle.won:
            end_reason = "win"
        else:
            end_reason = "loss"

        rows.append(
            (
                meta["run_id"],
                meta["train_steps"],
                meta["train_episodes"],
                opponent,
                int(deterministic),
                eval_seed,
                battle_idx,
                int(bool(battle.won)),
                battle.turn,
                round(reward, 4),
                end_reason,
                int(battle.used_tera),
                BROUGHT_SIZE - sum(p.fainted for p in battle.team.values()),
                BROUGHT_SIZE - sum(p.fainted for p in battle.opponent_team.values()),
            )
        )
        obs, _ = env.reset()
    return rows


def main(battles: int, final_battles: int, eval_seed: int, modes: list):
    con = connect()
    done = done_cells(con)
    found = checkpoints(MODEL_DIR)
    if not found:
        print(f"no checkpoints in {MODEL_DIR}")
        return

    # training only writes .zip/.json to disk, so the runs table gets filled in
    # here from the checkpoint sidecars. found is sorted by train_steps, so the
    # last checkpoint of a run wins and its wall clock is the total.
    for _, meta in found:
        if "seed" not in meta:
            print(f"no run metadata for {meta['run_id']}, checkpoint predates it")
            continue
        add_run(
            con,
            (
                meta["run_id"],
                meta["seed"],
                meta["commit_sha"],
                meta["feature_version"],
                meta["battle_format"],
                meta["team_id"],
                json.dumps(meta["hyperparams"]),
                meta["device"],
                meta["wall_clock_s"],
            ),
        )

    last_steps = found[-1][1]["train_steps"]
    for opponent in OPPONENTS:
        # one env per opponent, reused across every checkpoint: creating an env per
        # cell leaks showdown usernames and the sweep dies on "name taken"
        env = make_eval_env(opponent)
        try:
            for model_path, meta in found:
                n_battles = final_battles if meta["train_steps"] == last_steps else battles
                pending = [
                    d for d in modes
                    if done.get((meta["run_id"], meta["train_steps"], opponent, int(d)), 0) < n_battles
                ]
                if not pending:
                    print(f"skip {meta['run_id']} @{meta['train_steps']:>8} vs {opponent}")
                    continue
                model = MaskablePPO.load(model_path, env=env, device="auto")
                for deterministic in pending:
                    rows = play(model, env, meta, opponent, deterministic, eval_seed, n_battles)
                    add_battles(con, rows)
                    wins = sum(r[7] for r in rows)
                    print(
                        f"{meta['run_id']} @{meta['train_steps']:>8} vs {opponent:<22} "
                        f"det={int(deterministic)}  {wins}/{len(rows)}"
                    )
        finally:
            env.close()
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--final-battles", type=int, default=2000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="also evaluate the sampled policy")
    args = parser.parse_args()
    main(args.battles, args.final_battles, args.eval_seed, [True, False] if args.stochastic else [True])
