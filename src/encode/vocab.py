from poke_env.battle import PokemonGender
from poke_env.battle.effect import Effect
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.move import MoveCategory
from poke_env.battle.target import Target
from poke_env.data import GenData, to_id_str


FEATURE_VERSION = 1
GEN_DATA = GenData.from_gen(9)

ITEMS = (
    "choiceband", "choicespecs", "choicescarf", "lifeorb", "expertbelt", "assaultvest",
    "rockyhelmet", "eviolite", "leftovers", "shellbell", "safetygoggles", "covertcloak",
    "clearamulet", "focussash", "focusband", "airballoon", "boosterenergy", "loadeddice",
    "punchingglove", "abilityshield", "mirrorherb", "mentalherb", "powerherb", "whiteherb",
    "weaknesspolicy", "throatspray", "blunderpolicy", "roomservice", "ejectbutton",
    "ejectpack", "redcard", "stickybarb", "blacksludge", "toxicorb", "flameorb", "metronome",
    "kingsrock", "quickclaw", "brightpowder", "scopelens", "razorclaw", "widelens", "zoomlens",
    "lightclay", "terrainextender", "heatrock", "damprock", "smoothrock", "icyrock",
    "electricseed", "grassyseed", "mistyseed", "psychicseed",
    "sitrusberry", "lumberry", "chestoberry", "oranberry", "figyberry", "wikiberry",
    "magoberry", "aguavberry", "iapapaberry", "salacberry", "liechiberry", "starfberry",
    "micleberry", "custapberry", "occaberry", "passhoberry", "wacanberry", "rindoberry",
    "yacheberry", "chopleberry", "kebiaberry", "shucaberry", "cobaberry", "payapaberry",
    "tangaberry", "chartiberry", "kasibberry", "habanberry", "colburberry", "babiriberry",
    "chilanberry", "roseliberry",
    "wellspringmask", "hearthflamemask", "cornerstonemask", "rustedsword", "rustedshield",
    "adamantcrystal", "lustrousglobe", "griseouscore", "leek", "thickclub", "lightball",
    "luckypunch", "metalpowder", "souldew", "deepseatooth", "deepseascale", "berryjuice",
    "normalgem", "utilityumbrella", "protectivepads", "bigroot", "bindingband", "gripclaw",
    "ironball", "laggingtail", "machobrace", "poweranklet", "ringtarget", "shedshell",
    "absorbbulb", "cellbattery", "luminousmoss", "snowball",
)
SPECIES_NUM = {k: v["num"] for k, v in GEN_DATA.pokedex.items()}
ABILITY_NUM = {a: i + 1 for i, a in enumerate(sorted({to_id_str(a) for e in GEN_DATA.pokedex.values() for a in e.get("abilities", {}).values()} - {""}))}
ITEM_NUM = {k: i + 1 for i, k in enumerate(ITEMS)}


BASE_STAT_CAP = 255
REAL_STAT_CAP = {"hp": 714, "atk": 660, "def": 660, "spa":660, "spd": 660, "spe": 660}
BOOST_KEYS = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")
TYPES = ("Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "Stellar")
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
EFFECT_DURATION_CAPS = {
    Effect.CONFUSION: None,
    Effect.TAUNT: 3.0,
    Effect.LEECH_SEED: None,
    Effect.SUBSTITUTE: None,
    Effect.ENCORE: 3.0,
    Effect.DISABLE: 4.0,
    Effect.YAWN: None,
    Effect.INGRAIN: None,
    Effect.AQUA_RING: None,
    Effect.MAGNET_RISE: 5.0,
    Effect.SALT_CURE: None,
    Effect.LOCKED_MOVE: None,
    Effect.PARTIALLY_TRAPPED: None,
    Effect.TRAPPED: None,
    Effect.FLINCH: None,
    Effect.FOLLOW_ME: None,
    Effect.RAGE_POWDER: None,
    Effect.HELPING_HAND: None,
    Effect.COMMANDER: None,
    Effect.DRAGON_CHEER: None
}
PERISH_EFFECTS = (Effect.PERISH3, Effect.PERISH2, Effect.PERISH1, Effect.PERISH0)
PROTECT_COUNTER_CAP = 3.0

WEATHER_DURATION_CAPS = {
    Weather.SUNNYDAY: 8.0,
    Weather.RAINDANCE: 8.0,
    Weather.SANDSTORM: 8.0,
    Weather.SNOWSCAPE: 8.0,
    Weather.HAIL: 8.0,
    Weather.DESOLATELAND: None,
    Weather.PRIMORDIALSEA: None,
    Weather.DELTASTREAM: None,
}

FIELD_DURATION_CAPS = {
    Field.ELECTRIC_TERRAIN: 8.0,
    Field.GRASSY_TERRAIN: 8.0,
    Field.MISTY_TERRAIN: 8.0,
    Field.PSYCHIC_TERRAIN: 8.0,
    Field.TRICK_ROOM: 5.0,
    Field.GRAVITY: 5.0,
    Field.MAGIC_ROOM: 5.0,
    Field.WONDER_ROOM: 5.0,
}

SIDE_CONDITION_STACKABLE_CAPS = {
    SideCondition.SPIKES: 3.0,
    SideCondition.TOXIC_SPIKES: 2.0,
}

SIDE_CONDITION_DURATION_CAPS = {
    SideCondition.STEALTH_ROCK: None,
    SideCondition.STICKY_WEB: None,
    SideCondition.REFLECT: 8.0,
    SideCondition.LIGHT_SCREEN: 8.0,
    SideCondition.AURORA_VEIL: 8.0,
    SideCondition.TAILWIND: 4.0,
    SideCondition.SAFEGUARD: 5.0,
    SideCondition.MIST: 5.0,
    SideCondition.LUCKY_CHANT: 5.0,
    SideCondition.WIDE_GUARD: None,
    SideCondition.QUICK_GUARD: None,
    SideCondition.CRAFTY_SHIELD: None,
    SideCondition.MATBLOCK: None,
}

MOVE_CATEGORIES = (MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS)
MOVE_TARGETS = tuple(Target)
MOVE_FLAGS = (
    "contact",
    "protect",
    "reflectable",
    "bypasssub",
    "sound",
    "punch",
    "slicing",
    "bullet",
    "bite",
    "pulse",
    "wind",
    "powder",
    "heal",
    "charge",
    "recharge",
    "dance",
    "distance",
)
