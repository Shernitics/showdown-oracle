import numpy as np
from poke_env.battle import Pokemon

from encode.vocab import *
from encode.helpers import normalize

# cont - continuous features (float32); cat - categorical features (int64)
POKEMON_FEATURES_CONT = 110
POKEMON_FEATURES_CAT = 3


def encode_pokemon(pokemon: Pokemon | None, position: int | None):
    if pokemon is None:
        return {
            "cont": np.zeros(POKEMON_FEATURES_CONT, dtype=np.float32),
            "cat": {
                "species": np.zeros(1, dtype=np.int64),
                "items": np.zeros(1, dtype=np.int64),
                "ability": np.zeros(1, dtype=np.int64),
            }
        }

    cat = {
        "species": np.asarray([SPECIES_NUM.get(pokemon.species, 0)], dtype=np.int64),
        "items": np.asarray([ITEM_NUM.get(pokemon.item, 0)], dtype=np.int64),
        "ability": np.asarray([ABILITY_NUM.get(pokemon.ability, 0)], dtype=np.int64),
    }

    # position: left, right, unknown
    active = [float(position == 0), float(position == 1), float(position is None)]

    # pokemon characteristics (universal for this species)
    base_stats = [normalize(s, BASE_STAT_CAP) for s in pokemon.base_stats.values()]
    type = [float(type_name in {t.name.capitalize() for t in pokemon.types}) for type_name in TYPES]

    # stats
    current_hp_fraction = pokemon.current_hp_fraction if pokemon.revealed else 1.0
    stats_known = 1.0 if pokemon.stats.get("atk") is not None else 0.0
    stats = [normalize(v, REAL_STAT_CAP.get(s)) if stats_known else 0.0 for s, v in pokemon.stats.items()]
    boosts = [normalize(pokemon.boosts[k] + 6, 12) for k in BOOST_KEYS]

    # pokemon oriented
    level = normalize(pokemon.level, 100)
    gender = [1.0 if pokemon.gender is g else 0.0 for g in GENDERS]
    must_recharge = float(pokemon.must_recharge)
    first_turn = float(pokemon.first_turn)
    revealed = float(pokemon.revealed)
    brought = float(pokemon.selected_in_teampreview or pokemon.revealed)

    # terastallization
    tera = pokemon.tera_type
    tera_type = [float(tera is not None and t == tera.name.capitalize()) for t in TYPES]
    is_terastallized = float(pokemon.is_terastallized)

    # status / effects
    current_status = pokemon.status.name if pokemon.status else "NONE"
    status_name = [float(n == current_status) for n in STATUS_DURATION_CAPS]
    status_duration_max = STATUS_DURATION_CAPS.get(current_status)
    status_duration = normalize(pokemon.status_counter, status_duration_max) if status_duration_max else 0.0

    pokemon_effects = pokemon.effects
    current_effects = [float(e in pokemon_effects) for e in EFFECT_DURATION_CAPS]
    effects_duration = [normalize(pokemon_effects.get(e, 0), c) for e, c in EFFECT_DURATION_CAPS.items() if c is not None]
    perish_effects = [(p in pokemon_effects) for p in PERISH_EFFECTS]
    protect_counter = normalize(pokemon.protect_counter, PROTECT_COUNTER_CAP)


    cont = np.asarray(
        [
            1.0, *active,

            # pokemon characteristics (universal for this species)
            *base_stats, *type,

            # combat relevance

            # stats
            current_hp_fraction, stats_known, *stats, *boosts,

            # pokemon oriented
            level, *gender, must_recharge, first_turn, revealed, brought,

            # terastallization
            *tera_type, is_terastallized,

            # status / volatile
            *status_name, status_duration, *current_effects, *effects_duration, *perish_effects, protect_counter

        ]
    , dtype=np.float32)

    return {
        "cont": cont,
        "cat": cat
    }
