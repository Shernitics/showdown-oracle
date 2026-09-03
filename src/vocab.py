import json
from pathlib import Path

from poke_env.battle import PokemonGender, Status
from poke_env.battle.effect import Effect

with open(Path(__file__).resolve().parent.parent / "data" / "pokedex.json", "r", encoding="utf-8") as f:
    pokedex = json.load(f)

with open(Path(__file__).resolve().parent.parent / "data" / "moves.json", "r", encoding="utf-8") as f:
    moves = json.load(f)

with open(Path(__file__).resolve().parent.parent / "data" / "abilities.json", "r", encoding="utf-8") as f:
    abilities = json.load(f)

with open(Path(__file__).resolve().parent.parent / "data" / "items.json", "r", encoding="utf-8") as f:
    items = json.load(f)

# {'bulbasaur':{"num":1, "name": "Bulbasaur", "types":["Grass", "Poison"],"genderRatio":{"M":0.875, "F":0.125}, "baseStats":{"hp":45, "atk":49, "def":49, "spa":65, "spd":65, "spe":45}, "abilities":{"0": "Overgrow", "H": "Chlorophyll"}, "heightm":0.7, "weightkg":6.9, "color": "Green", "evos":["Ivysaur"], "eggGroups":["Monster", "Grass"], "tier": "LC"}

TYPES = ("Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy")
GENDERS = (PokemonGender.MALE, PokemonGender.FEMALE, PokemonGender.NEUTRAL)
STATUS_DURATION_CAPS = {
    "NONE": None,
    "BRN": None,
    "FRZ": None,
    "PAR": None,
    "PSN": None,
    "SLP": 3.0,
    "TOX": 15.0,
    "FNT": None,
}