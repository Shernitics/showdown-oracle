import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from encode.battle import ENVIRONMENT_FEATURES_CONT
from encode.moves import MOVE_FEATURES_CONT
from encode.pokemon import POKEMON_FEATURES_CONT
from encode.state import MOVE_SLOTS, TEAM_SIZE
from encode.vocab import ABILITY_NUM, ITEM_NUM, SPECIES_NUM

SLOTS = TEAM_SIZE * 2
MAX_MOVE_NUM = 1000
EMB_DIM = 16
MON_DIM = 128


class VGCExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super().__init__(
            observation_space, SLOTS * MON_DIM + ENVIRONMENT_FEATURES_CONT
        )

        self.species = nn.Embedding(len(SPECIES_NUM) + 1, EMB_DIM)
        self.item = nn.Embedding(len(ITEM_NUM) + 1, EMB_DIM)
        self.ability = nn.Embedding(len(ABILITY_NUM) + 1, EMB_DIM)
        self.move = nn.Embedding(MAX_MOVE_NUM + 1, EMB_DIM)

        mon_inputs = (
            POKEMON_FEATURES_CONT
            + 3 * EMB_DIM
            + MOVE_SLOTS * (MOVE_FEATURES_CONT + EMB_DIM)
        )
        self.mon = nn.Sequential(nn.Linear(mon_inputs, MON_DIM), nn.ReLU())

    def forward(self, obs):
        batch = obs["pokemon_cont"].shape[0]

        move_ids = obs["moves_cat"].long().squeeze(-1).clamp(0, MAX_MOVE_NUM)
        moves = torch.cat([obs["moves_cont"], self.move(move_ids)], dim=-1)
        moves = moves.reshape(batch, SLOTS, -1)

        mons = torch.cat(
            [
                obs["pokemon_cont"],
                self.species(obs["species"].long().squeeze(-1)),
                self.item(obs["item"].long().squeeze(-1)),
                self.ability(obs["ability"].long().squeeze(-1)),
                moves,
            ],
            dim=-1,
        )
        mons = self.mon(mons).reshape(batch, -1)

        return torch.cat([mons, obs["battle_cont"]], dim=-1)
