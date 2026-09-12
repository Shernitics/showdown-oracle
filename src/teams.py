from poke_env.teambuilder import ConstantTeambuilder

FORMAT = "gen9vgc2025regi"

PASTE = """
Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Fake Out
- Knock Off
- Parting Shot
- Will-O-Wisp

Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
Tera Type: Fire
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Grassy Glide
- Wood Hammer
- U-turn
- Fake Out

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Thunderbolt

Amoonguss @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Water
EVs: 252 HP / 172 Def / 84 SpD
Calm Nature
- Spore
- Rage Powder
- Pollen Puff
- Protect

Urshifu-Rapid-Strike @ Focus Sash
Ability: Unseen Fist
Level: 50
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- Protect

Landorus-Therian @ Choice Scarf
Ability: Intimidate
Level: 50
Tera Type: Flying
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- U-turn
- Tera Blast
"""

TEAM = ConstantTeambuilder(PASTE)

TEAM_ID = "mirror-v1"
