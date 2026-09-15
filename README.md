# VGC Oracle

> A reinforced-learning agent that plays competitive Pokémon Showdown (Doubles Generation 9) and recommends the best moves.

## Status
Early development.

## What it does
Given a battle state, it recommends chains of strongest actions. It learns to play by **self-play reinforcement learning** rather than handwritten rules.

## How it works
- Trains against a local **Pokémon Showdown** server **poke-env**.
- A **PyTorch** network (the "brain") is trained and the result is a file, `model.pt`, which scores each legal move from the current board.

## Results
| Opponent            | Win rate |
|---------------------|----------|
| RandomPlayer        | TBD      |
| MaxBasePowerPlayer  | TBD      |

## Tech
Python · PyTorch · poke-env · Pokémon Showdown · Stable-Baselines3

# Starting Pokemon Showdown server
```commandline
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```