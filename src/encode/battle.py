import numpy as np
from poke_env.battle import DoubleBattle

from encode.vocab import *
from encode.helpers import normalize

ENVIRONMENT_FEATURES_CONT = 81
BROUGHT_SIZE = 4

def encode_battle(battle: DoubleBattle):

    turn = battle.turn

    turn_norm = normalize(turn, 100)
    pokemon_remaining = normalize(BROUGHT_SIZE - sum(1 for p in battle.team.values() if p.fainted), BROUGHT_SIZE)
    pokemon_remaining_opp = normalize(BROUGHT_SIZE - sum(1 for p in battle.opponent_team.values() if p.fainted), BROUGHT_SIZE)

    # weather
    weather = [float(w in battle.weather) for w in WEATHER_DURATION_CAPS.keys()]
    current_weather = list(battle.weather)[0] if battle.weather else None
    weather_duration_max = WEATHER_DURATION_CAPS.get(current_weather)
    weather_duration = normalize(turn - battle.weather[current_weather], weather_duration_max) if weather_duration_max else 0.0

    # field
    field = [float(f in battle.fields) for f in FIELD_DURATION_CAPS.keys()]
    field_duration = [normalize(turn - battle.fields.get(f, turn), c) for f, c in FIELD_DURATION_CAPS.items()]

    # side conditions
    side_conditions = [float(s in battle.side_conditions) for s in SIDE_CONDITION_DURATION_CAPS.keys()]
    side_conditions_duration = [normalize(turn - battle.side_conditions.get(s, turn), c) for s, c in SIDE_CONDITION_DURATION_CAPS.items() if c is not None]
    side_conditions_opp = [float(s in battle.opponent_side_conditions) for s in SIDE_CONDITION_DURATION_CAPS.keys()]
    side_conditions_duration_opp = [normalize(turn - battle.opponent_side_conditions.get(s, turn), c) for s, c in SIDE_CONDITION_DURATION_CAPS.items() if c is not None]

    # side conditions (stackable)
    side_conditions_stackable = [normalize(battle.side_conditions.get(s, 0), c) for s, c in SIDE_CONDITION_STACKABLE_CAPS.items()]
    side_conditions_stackable_opp = [normalize(battle.opponent_side_conditions.get(s, 0), c) for s, c in SIDE_CONDITION_STACKABLE_CAPS.items()]

    # misc
    can_tera = battle.can_tera
    used_tera = [battle.used_tera, battle.opponent_used_tera]
    force_switch = battle.force_switch
    trapped = battle.trapped
    reviving = battle.reviving

    cont = np.asarray(
        [

            # battle
            turn_norm, pokemon_remaining, pokemon_remaining_opp,

            # weather
            *weather, weather_duration,

            # field
            *field, *field_duration,

            # side conditions
            *side_conditions, *side_conditions_duration, *side_conditions_opp, *side_conditions_duration_opp,

            # side conditions (stackable)
            *side_conditions_stackable, *side_conditions_stackable_opp,

            # misc
            *can_tera, *used_tera, *force_switch, *trapped, reviving,

        ]
    , dtype=np.float32)

    return {"cont": cont}
