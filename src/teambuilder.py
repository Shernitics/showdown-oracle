import random
from pathlib import Path

from poke_env.teambuilder import Teambuilder


class VGCTeams(Teambuilder):

    def __init__(self, directory, seed=0):
        self.teams = [
            self.join_team(self.parse_showdown_team(p.read_text(encoding="utf-8")))
            for p in sorted(Path(directory).glob("*.txt"))
        ]
        random.Random(seed).shuffle(self.teams)
        self.index = 0

    def yield_team(self):
        team = self.teams[self.index % len(self.teams)]
        self.index += 1
        return team
