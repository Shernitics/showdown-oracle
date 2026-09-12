from pathlib import Path

from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer
from sb3_contrib import MaskablePPO

from env import VGCEnv
from extractor import VGCExtractor
from teams import FORMAT, TEAM
from wrapper import MaskedSingleAgent

TOTAL_STEPS = 50_000
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "vgc"


def make_env():
    env = VGCEnv(
        battle_format=FORMAT,
        team=TEAM,
        strict=False,
        choose_on_teampreview=False,
    )
    opponent = RandomPlayer(
        battle_format=FORMAT, team=TEAM, start_listening=False
    )
    return MaskedSingleAgent(SingleAgentWrapper(env, opponent))


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    env = make_env()
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={"features_extractor_class": VGCExtractor},
        verbose=1,
    )
    model.learn(total_timesteps=TOTAL_STEPS)
    model.save(MODEL_PATH)

    print(f"steps {env.n_steps}  repairs {env.n_repairs}  rate {env.repair_rate:.3f}")


if __name__ == "__main__":
    main()
