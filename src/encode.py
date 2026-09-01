import numpy as np
from poke_env.battle import DoubleBattle, Pokemon, Move

POKEMON_FEATURES_CONT = 1
POKEMON_FEATURES_CAT = 1


def encode_pokemon(pokemon: Pokemon | None):
    if pokemon is None:
        return {
            "cont": np.zeros(POKEMON_FEATURES_CONT, dtype=np.float32),
            "cat": {
                "species": np.zeros(1, dtype=np.int64),
                "items": np.zeros(1, dtype=np.int64),
                "ability": np.zeros(1, dtype=np.int64),
            }
        }
    