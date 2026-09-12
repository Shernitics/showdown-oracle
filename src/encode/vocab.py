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
    "abilityshield", "absorbbulb", "adamantcrystal", "adamantorb", "adrenalineorb",
    "aguavberry", "airballoon", "apicotberry", "aspearberry", "assaultvest", "auspiciousarmor",
    "babiriberry", "beastball", "berrysweet", "bignugget", "bigroot", "bindingband",
    "blackbelt", "blackglasses", "blacksludge", "blunderpolicy", "boosterenergy", "bottlecap",
    "brightpowder", "cellbattery", "charcoal", "chartiberry", "cheriberry", "chestoberry",
    "chilanberry", "chippedpot", "choiceband", "choicescarf", "choicespecs", "chopleberry",
    "clearamulet", "cloversweet", "cobaberry", "colburberry", "cornerstonemask", "covertcloak",
    "crackedpot", "custapberry", "damprock", "dawnstone", "destinyknot", "diveball",
    "dracoplate", "dragonfang", "dragonscale", "dreadplate", "dreamball", "dubiousdisc",
    "duskball", "duskstone", "earthplate", "ejectbutton", "ejectpack", "electirizer",
    "electricseed", "enigmaberry", "eviolite", "expertbelt", "fairyfeather", "fastball",
    "figyberry", "firestone", "fistplate", "flameorb", "flameplate", "floatstone",
    "flowersweet", "focusband", "focussash", "friendball", "galaricacuff", "galaricawreath",
    "ganlonberry", "goldbottlecap", "grassyseed", "greatball", "grepaberry", "gripclaw",
    "griseouscore", "griseousorb", "habanberry", "hardstone", "healball", "hearthflamemask",
    "heatrock", "heavyball", "heavydutyboots", "hondewberry", "iapapaberry", "icestone",
    "icicleplate", "icyrock", "insectplate", "ironball", "ironplate", "jabocaberry",
    "kasibberry", "kebiaberry", "keeberry", "kelpsyberry", "kingsrock", "laggingtail",
    "lansatberry", "leafstone", "leftovers", "leppaberry", "levelball", "liechiberry",
    "lifeorb", "lightball", "lightclay", "loadeddice", "loveball", "lovesweet", "lumberry",
    "luminousmoss", "lureball", "lustrousglobe", "lustrousorb", "luxuryball", "magmarizer",
    "magnet", "magoberry", "maliciousarmor", "marangaberry", "masterball", "masterpieceteacup",
    "meadowplate", "mentalherb", "metalalloy", "metalcoat", "metronome", "micleberry",
    "mindplate", "miracleseed", "mirrorherb", "mistyseed", "moonball", "moonstone",
    "muscleband", "mysticwater", "nestball", "netball", "nevermeltice", "normalgem",
    "occaberry", "oranberry", "ovalstone", "passhoberry", "payapaberry", "pechaberry",
    "persimberry", "petayaberry", "pixieplate", "poisonbarb", "pokeball", "pomegberry",
    "poweranklet", "powerband", "powerbelt", "powerbracer", "powerherb", "powerlens",
    "powerweight", "premierball", "prettyfeather", "prismscale", "protectivepads", "protector",
    "psychicseed", "punchingglove", "qualotberry", "quickball", "quickclaw", "rarebone",
    "rawstberry", "razorclaw", "razorfang", "reapercloth", "redcard", "repeatball",
    "ribbonsweet", "rindoberry", "ringtarget", "rockyhelmet", "roomservice", "roseliberry",
    "rowapberry", "rustedshield", "rustedsword", "safariball", "safetygoggles", "salacberry",
    "scopelens", "sharpbeak", "shedshell", "shellbell", "shinystone", "shucaberry", "silkscarf",
    "silverpowder", "sitrusberry", "skyplate", "smoothrock", "snowball", "softsand", "souldew",
    "spelltag", "splashplate", "spookyplate", "sportball", "starfberry", "starsweet",
    "stickybarb", "stoneplate", "strawberrysweet", "sunstone", "sweetapple", "syrupyapple",
    "tamatoberry", "tangaberry", "tartapple", "terrainextender", "throatspray", "thunderstone",
    "timerball", "toxicorb", "toxicplate", "twistedspoon", "ultraball", "unremarkableteacup",
    "upgrade", "utilityumbrella", "wacanberry", "waterstone", "weaknesspolicy",
    "wellspringmask", "whiteherb", "widelens", "wikiberry", "wiseglasses", "yacheberry",
    "zapplate", "zoomlens",
)

ITEM_UNKNOWN, ITEM_NONE = 1, 2
SPECIES_NUM = {k: i + 1 for i, k in enumerate(sorted(GEN_DATA.pokedex))}
ABILITY_NUM = {a: i + 1 for i, a in enumerate(sorted({to_id_str(a) for e in GEN_DATA.pokedex.values() for a in e.get("abilities", {}).values()} - {""}))}
ITEM_NUM = {
    GEN_DATA.UNKNOWN_ITEM: ITEM_UNKNOWN,
    None: ITEM_NONE,
    **{k: i + 3 for i, k in enumerate(ITEMS)},
}


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
