import numpy as np
from poke_env.battle import DoubleBattle

from encode.pokemon import encode_pokemon
from encode.moves import encode_move
from encode.battle import encode_battle


TEAM_SIZE = 6
MOVE_SLOTS = 4

def encode_state(battle: DoubleBattle):

    live = {p.base_species: p for p in battle.opponent_team.values()}
    preview = battle.teampreview_opponent_team

    if preview:
        opp = [live.get(p.base_species, p) for p in preview][:TEAM_SIZE]
    else:
        opp = list(battle.opponent_team.values())[:TEAM_SIZE]

    own = list(battle.team.values())[:TEAM_SIZE]
    own += [None] * (TEAM_SIZE - len(own))
    opp += [None] * (TEAM_SIZE - len(opp))

    slots = own + opp

    positions = {}
    for i, p in enumerate(battle.active_pokemon):
        if p is not None:
            positions[id(p)] = i

    for i, p in enumerate(battle.opponent_active_pokemon):
        if p is not None:
            positions[id(p)] = i

    # Pokemon (player, opp) (asc)
    pokemon_enc = [encode_pokemon(p, positions.get(id(p)) if p else None) for p in slots]
    moves_enc = []
    for p in slots:
        m = list(p.moves.values())[:MOVE_SLOTS] if p else []
        m += [None] * (MOVE_SLOTS - len(m))
        moves_enc.append([encode_move(x) for x in m])
    battle_enc = encode_battle(battle)

    return {
        "pokemon_cont": np.stack([p["cont"] for p in pokemon_enc]),
        "pokemon_cat": {k: np.stack([p["cat"][k] for p in pokemon_enc]) for k in ("species", "items", "ability")},
        "moves_cont": np.stack([np.stack([m["cont"] for m in row]) for row in moves_enc]),
        "moves_cat": np.stack([np.stack([m["cat"]["name"] for m in row]) for row in moves_enc]),
        "battle_cont": battle_enc["cont"],
    }
