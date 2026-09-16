from shutil import move

import torch
import torch.nn as nn
from poke_env.battle import pokemon
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from encode.state import TEAM_SIZE, MOVE_SLOTS
from encode.pokemon import POKEMON_FEATURES_CONT
from encode.moves import MOVE_FEATURES_CONT
from encode.battle import ENVIRONMENT_FEATURES_CONT
from encode.vocab import SPECIES_NUM, ITEM_NUM, ABILITY_NUM, MOVE_NUM

SLOTS = TEAM_SIZE * 2
EMBEDDING_DIM = 16 # length of numbers to categorize
POKEMON_HIDDEN = 128


class VGCExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):

        # 108 = 92 + 16
        move_width = MOVE_FEATURES_CONT + EMBEDDING_DIM

        # 590 = 110 + (3 * 16) {species, items, abilities} + (4 * 108)
        pokemon_width = POKEMON_FEATURES_CONT + 3 * EMBEDDING_DIM + MOVE_SLOTS * move_width

        # 1617 = (12 * 128) + 81
        features_dim = SLOTS * POKEMON_HIDDEN + ENVIRONMENT_FEATURES_CONT

        super(VGCExtractor, self).__init__(observation_space, features_dim)
        self.species = nn.Embedding(len(SPECIES_NUM) + 1, EMBEDDING_DIM)
        self.item = nn.Embedding(len(ITEM_NUM) + 1, EMBEDDING_DIM)
        self.ability = nn.Embedding(len(ABILITY_NUM) + 1, EMBEDDING_DIM)
        self.move = nn.Embedding(len(MOVE_NUM) + 1, EMBEDDING_DIM)
        self.pokemon = nn.Sequential(
            nn.Linear(pokemon_width, POKEMON_HIDDEN), # (in, out)
            nn.ReLU(), # max(0, x)
        ) # can try LeakyReLU()

    def forward(self, observations):

        species = self.species(observations["species"].squeeze(-1).long())
        ability = self.ability(observations["ability"].squeeze(-1).long())
        item = self.item(observations["item"].squeeze(-1).long())
        move = self.move(observations["moves_cat"].squeeze(-1).long())

        moves = torch.cat([observations["moves_cont"], move], dim=-1).flatten(2)
        pokemon = torch.cat([observations["pokemon_cont"], species, item, ability, moves], dim=-1)
        pokemon = self.pokemon(pokemon).flatten(1)

        return torch.cat([pokemon, observations["battle_cont"]], dim=-1)
