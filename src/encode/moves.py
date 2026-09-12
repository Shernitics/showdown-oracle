import numpy as np
from poke_env.battle import Move

from encode.vocab import *
from encode.helpers import normalize

MOVE_FEATURES_CONT = 92
MOVE_FEATURES_CAT = 1

def encode_move(move: Move):

    if move is None:
        return {
            "cont": np.zeros(MOVE_FEATURES_CONT, dtype=np.float32),
            "cat": {
                "name": np.zeros(1, dtype=np.int64),
            }
        }

    cat = {
        "name": np.asarray([move.entry.get("num", 0)], dtype=np.int64),
    }


    # move characteristics
    base_power = normalize(move.base_power, 250)
    accuracy = move.accuracy
    priority = normalize(move.priority + 7, 12)
    max_pp = move.max_pp
    pp = normalize(move.current_pp, max_pp) if max_pp else 0.0

    # poke-env rebuilds these on every access, so read them once
    move_category = move.category
    type_name = move.type.name.capitalize()
    move_target = move.target
    move_flags = move.flags

    category = [float(move_category is c) for c in MOVE_CATEGORIES]
    type = [float(t == type_name) for t in TYPES]
    target = [float(move_target is t) for t in MOVE_TARGETS]
    flags = [float(f in move_flags) for f in MOVE_FLAGS]

    # stat changes
    move_boosts = move.boosts or {}
    move_self_boost = move.self_boost or {}
    boosts = [normalize(move_boosts.get(k, 0) + 6, 12) for k in BOOST_KEYS]
    self_boost = [normalize(move_self_boost.get(k, 0) + 6, 12) for k in BOOST_KEYS]

    # status
    move_status = move.status
    current_status = move_status.name if move_status is not None else None
    status = [float(n == current_status) for n in STATUS_DURATION_CAPS]

    # damage profile
    expected_hits = normalize(move.expected_hits, 5)
    crit_ratio = normalize(move.crit_ratio, 6)
    drain = move.drain
    recoil = move.recoil
    heal = move.heal

    # behavior
    force_switch = float(move.force_switch)
    self_switch = float(bool(move.self_switch))
    breaks_protect = float(move.breaks_protect)
    is_protect_move = float(move.is_protect_move)
    ignore_ability = float(move.ignore_ability)
    thaws_target = float(move.thaws_target)


    cont = np.asarray(
        [
            1.0,

            # move characteristics
            base_power, accuracy, priority, pp, *category, *type, *target, *flags,

            # stat changes
            *boosts, *self_boost,

            # status
            *status,

            # damage profile
            expected_hits, crit_ratio, drain, recoil, heal,

            # behaviour
            force_switch, self_switch, breaks_protect, is_protect_move, ignore_ability, thaws_target,

        ], dtype=np.float32
    )

    return {
        "cont": cont,
        "cat": cat
    }
