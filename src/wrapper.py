"""
The wrapper is a layer with 2 important roles:
    - Splits observation and action mask from the environment. The observation is sent downwards to the agent whereas the action mask remains.
    - Repairs if both Pokémon slots use the same move which cannot be used together such as:
        - both slots terastallize
        - both slots switch to the same Pokémon
        - both slots pass
"""

