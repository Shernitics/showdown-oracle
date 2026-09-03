"""
Continuous
- Exists, Species, Real stats (6), Stats known flag (Unknown if opp), Base Stats (6), HP fraction
- Boost (7: atk/def/spa/spd/spe/accuracy/evasion)
- Level, Gender, Active
- Must recharge (hyper beam / Giga impact lock)
- First Turn (For Fake Out / First Impression)
- Revealed

Status / Volatiles
- Status one-hot + duration (STATUS_DURATION_CAPS)
- Volatile effects + duration (EFFECT_DURATION_CAPS)
    (confusion, taunt, leech seed, sub, encore, disable,
    yawn, ingrain, aqua ring, magnet rise, salt cure,
    locked move, partial trap, perish counts)
- Protect Counter decay

Typing / Tera
- Current Type (18)
- Tera type + is terastallized
"""