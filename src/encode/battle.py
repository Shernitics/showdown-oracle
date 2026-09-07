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

# -----------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from poke_env.battle import DoubleBattle, Pokemon
    from poke_env.battle.weather import Weather
    from poke_env.battle.field import Field
    from poke_env.battle.side_condition import SideCondition
    from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

    def mk(paste):
        m = Pokemon(gen=9, teambuilder=TeambuilderPokemon.from_showdown(paste))
        m._active = True          # required: active_pokemon filters out inactive mons
        m._revealed = True
        return m

    battle = DoubleBattle("battle-gen9vgc2024-1", "test-player",
                          logging.getLogger("test"), gen=9)

    battle._player_role = "p1"    # required, or active_pokemon raises ValueError
    battle._turn = 12

    # NOTE: values are the TURN THE CONDITION STARTED (elapsed = turn - value),
    # except SPIKES / TOXIC_SPIKES, whose values are LAYER COUNTS.
    battle._weather = {Weather.SUNNYDAY: 9}                    # 3 turns elapsed
    battle._fields = {Field.TRICK_ROOM: 10, Field.GRASSY_TERRAIN: 8}
    battle._side_conditions = {
        SideCondition.STEALTH_ROCK: 3,   # start turn
        SideCondition.SPIKES: 2,         # 2 LAYERS
        SideCondition.REFLECT: 9,        # start turn
    }
    battle._opponent_side_conditions = {
        SideCondition.TAILWIND: 11,
        SideCondition.TOXIC_SPIKES: 1,   # 1 LAYER
    }
    battle._can_tera = [True, False]

    flutter = mk("""Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Moonblast""")

    rillaboom = mk("""Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
Tera Type: Fire
EVs: 252 HP / 252 Atk
Adamant Nature
- Fake Out""")

    incin = Pokemon(gen=9, species="incineroar");  incin._active = True
    bolt  = Pokemon(gen=9, species="ragingbolt");  bolt._active = True

    # keys MUST be "<role>a" / "<role>b"
    battle._active_pokemon = {"p1a": flutter, "p1b": rillaboom}
    battle._opponent_active_pokemon = {"p2a": incin, "p2b": bolt}
    battle._team = {"p1: Flutter Mane": flutter, "p1: Rillaboom": rillaboom}
    battle._opponent_team = {"p2: Incineroar": incin, "p2: Raging Bolt": bolt}

    battle._player_role = "p1"
    battle._team_size = {"p1": 4, "p2": 4}     # <-- add this line
    battle._turn = 12

    print(encode_battle(battle)["cont"])