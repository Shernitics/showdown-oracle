import numpy as np
from poke_env.battle import Pokemon

from vocab import *

# cont - continuous features (float32); cat - categorical features (int64)
POKEMON_FEATURES_CONT = 1
POKEMON_FEATURES_CAT = 1

def normalize(value:int | float, cap: int | float) -> int | float:
    return min(max(float(value), 0.0), cap) / cap


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

    pokemon_ref = pokedex.get(pokemon.species)

    # "bulbasaur":{"num":1,"name":"Bulbasaur","types":["Grass","Poison"],"genderRatio":{"M":0.875,"F":0.125},"baseStats":{"hp":45,"atk":49,"def":49,"spa":65,"spd":65,"spe":45},"abilities":{"0":"Overgrow","H":"Chlorophyll"},"heightm":0.7,"weightkg":6.9,"color":"Green","evos":["Ivysaur"],"eggGroups":["Monster","Grass"],"tier":"LC"}

    cat = {
        "species": np.asarray(pokedex.get(pokemon.species, {}).get("num", 0), dtype=np.int64),
        "items": np.asarray(items.get(pokemon.item, {}).get("num", 0), dtype=np.int64),
        "ability": np.asarray(abilities.get(pokemon.ability, {}).get("num", 0), dtype=np.int64),
    }

    base_stats = [normalize(stat, 255) for stat in pokemon_ref.get("baseStats").values()]
    type = [float(type_name in {t.name.capitalize() for t in pokemon.types}) for type_name in TYPES]

    current_hp_fraction = pokemon.current_hp_fraction
    stats_known = pokemon.stats.get("atk") is not None
    stats = [normalize(stat, 255) if stats_known else 0.0 for stat in pokemon.stats.values()]
    boosts = [normalize(stat + 6, 12) for stat in pokemon.boosts.values()]

    level = normalize(pokemon.level, 100)
    gender = [1.0 if pokemon.gender is g else 0.0 for g in GENDERS]
    active = pokemon.active
    must_recharge = float(pokemon.must_recharge)
    first_turn = float(pokemon.first_turn)
    revealed = float(pokemon.revealed)

    tera = pokemon.tera_type
    tera_type = [float(tera is not None and type_name == tera.name.capitalize()) for type_name in TYPES]
    is_terastallized = float(pokemon.is_terastallized)

    current = pokemon.status.name if pokemon.status else "NONE"
    status_name = [float(name == current) for name in STATUS_DURATION_CAPS]
    status_duration_max = STATUS_DURATION_CAPS.get(current)
    status_duration = normalize(pokemon.status_counter, status_duration_max) if status_duration_max else 0.0

    cont = np.asarray(
        [
            1.0,

            # pokemon characteristics (universal for this species)
            *base_stats, *type,

            # combat relevance

            # stats
            current_hp_fraction, stats_known, *stats, *boosts,

            # pokemon oriented
            level, *gender, active, must_recharge, first_turn, revealed,

            # terastallization
            *tera_type, is_terastallized,

            # status / volatile
            *status_name, status_duration

        ]
    )

    return {
        "cont": cont,
        "cat": cat
    }



# -----------------------------------------------------------------
if __name__ == "__main__":
    from poke_env.battle import Pokemon
    from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

    # A real competitive set, in Showdown export format
    paste = """Great Tusk (M) @ Booster Energy
Ability: Protosynthesis
Level: 39
Tera Type: Steel
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Close Combat
- Ice Spinner
- Rapid Spin"""

    tb = TeambuilderPokemon.from_showdown(paste)
    example = Pokemon(gen=9, teambuilder=tb)

    print(encode_pokemon(example))