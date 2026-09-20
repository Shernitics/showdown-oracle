from pathlib import Path

from sb3_contrib import MaskablePPO
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment import SingleAgentWrapper


from env import VGCEnv
from wrapper import DoubleAgentWrapper
from extractor import VGCExtractor
from teambuilder import VGCTeams

FORMAT = "gen9vgc2025regi"
TEAM_DIR = Path(__file__).parent / "teams"
MODEL_DIR = Path(__file__).parent / "model"
TOTAL_TIMESTEPS = 2048

players = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]

def make_env():
    env = VGCEnv(
        battle_format=FORMAT,
        team=VGCTeams(TEAM_DIR),
        strict=False,
        choose_on_teampreview=True,
    )
    opponent = RandomPlayer(battle_format=FORMAT)
    return DoubleAgentWrapper(SingleAgentWrapper(env, opponent))

def main():
    env = make_env()
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={"features_extractor_class": VGCExtractor},
        verbose=1,
    )
    print("extractor:", type(model.policy.features_extractor).__name__)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        MODEL_DIR.mkdir(exist_ok=True)
        model.save(MODEL_DIR / "vgc")
        print(f"[saved]     {MODEL_DIR / 'vgc.zip'}")
        print(f"[repairs]   {env.n_repairs} of {env.n_steps} steps ({env.repair_rate:.2%})")

    finally:
        env.close()


if __name__ == "__main__":
    main()
