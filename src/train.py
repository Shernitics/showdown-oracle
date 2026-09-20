from pathlib import Path
from functools import partial

from poke_env import AccountConfiguration
from sb3_contrib import MaskablePPO
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment import SingleAgentWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env import VGCEnv
from wrapper import DoubleAgentWrapper
from extractor import VGCExtractor
from teambuilder import VGCTeams

FORMAT = "gen9vgc2025regi"
TEAM_DIR = Path(__file__).parent / "teams"
MODEL_DIR = Path(__file__).parent / "model"
STEPS_PER_ENV = 512
TOTAL_TIMESTEPS = 16896

PLAYERS = {
    RandomPlayer: "random",
    MaxBasePowerPlayer: "maxpower",
    SimpleHeuristicsPlayer: "heuristics",
}

def make_env(opponent_cls, seed):

    env = VGCEnv(
        battle_format=FORMAT,
        team=VGCTeams(TEAM_DIR, seed=seed),
        strict=False,
        choose_on_teampreview=True,
    )

    opponent = opponent_cls(
        battle_format=FORMAT,
        account_configuration=AccountConfiguration(f"opp-{PLAYERS[opponent_cls]}", None),
    )

    return Monitor(DoubleAgentWrapper(SingleAgentWrapper(env, opponent)))

def main():

    vec_env = SubprocVecEnv(
        [partial(make_env, cls, seed) for seed, cls in enumerate(PLAYERS)]
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=STEPS_PER_ENV,
        policy_kwargs={"features_extractor_class": VGCExtractor},
        verbose=1,
    )

    print("extractor:", type(model.policy.features_extractor).__name__)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        MODEL_DIR.mkdir(exist_ok=True)
        model.save(MODEL_DIR / "vgc")
        repairs = sum(vec_env.get_attr("n_repairs"))
        steps = sum(vec_env.get_attr("n_steps"))

        print(f"[saved]     {MODEL_DIR / 'vgc.zip'}")
        print(f"[repairs]   {repairs} of {steps} steps ({repairs / max(steps, 1):.2%})")

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
