import numpy as np
from gymnasium import spaces
from poke_env.environment import DoublesEnv
from poke_env.player import DefaultBattleOrder

from config import VICTORY_VALUE, FAINTED_VALUE, HP_VALUE, STATUS_VALUE
from encode.battle import ENVIRONMENT_FEATURES_CONT
from encode.moves import MOVE_FEATURES_CONT
from encode.pokemon import POKEMON_FEATURES_CONT
from encode.state import encode_state, TEAM_SIZE, MOVE_SLOTS
from encode.vocab import SPECIES_NUM, ITEM_NUM, ABILITY_NUM, MOVE_NUM

SLOTS = TEAM_SIZE * 2
BROUGHT_SIZE = 4


class VGCEnv(DoublesEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        observation = spaces.Dict(
            {
                "pokemon_cont": spaces.Box(0.0, 1.0, (SLOTS, POKEMON_FEATURES_CONT), np.float32),
                "species": spaces.Box(0, len(SPECIES_NUM), (SLOTS, 1), np.int64),
                "item": spaces.Box(0, len(ITEM_NUM), (SLOTS, 1), np.int64),
                "ability": spaces.Box(0, len(ABILITY_NUM), (SLOTS, 1), np.int64),
                "moves_cont": spaces.Box(0.0, 1.0, (SLOTS, MOVE_SLOTS, MOVE_FEATURES_CONT), np.float32),
                "moves_cat": spaces.Box(0, len(MOVE_NUM), (SLOTS, MOVE_SLOTS, 1), np.int64),
                "battle_cont": spaces.Box(0.0, 1.0, (ENVIRONMENT_FEATURES_CONT,), np.float32),
            }
        )
        self.observation_spaces = {a: observation for a in self.possible_agents}

    @staticmethod
    def action_to_order(action, battle, fake=False, strict=True):

        # poke-env rejects pass, pass, but when both slots can only pass it is
        # the one legal choice and the turn never advances without it
        if not battle.teampreview and action[0] == 0 and action[1] == 0:
            return DefaultBattleOrder()

        return DoublesEnv.action_to_order(
            action, battle, fake=fake or battle.teampreview, strict=strict
        )

    def embed_battle(self, battle):
        state = encode_state(battle)
        cat = state["pokemon_cat"]
        return {
            "pokemon_cont": state["pokemon_cont"],
            "species": cat["species"],
            "item": cat["items"],
            "ability": cat["ability"],
            "moves_cont": state["moves_cont"],
            "moves_cat": state["moves_cat"],
            "battle_cont": state["battle_cont"],
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            victory_value=VICTORY_VALUE,
            fainted_value=FAINTED_VALUE,
            hp_value=HP_VALUE,
            status_value=STATUS_VALUE,
            number_of_pokemons=BROUGHT_SIZE,
        )
