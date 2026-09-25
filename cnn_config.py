"""Edit this file to choose your rule bot and Battlesnake simulator.

Your bot only needs the usual info/start/move/end functions; Step 1 records
its decisions without requiring special recording code in the bot.
"""
from pathlib import Path

import main as RULE_BASED_AGENT  # Replace with your own module import.

PROJECT_DIR = Path(__file__).resolve().parent
ENGINE_PATH = PROJECT_DIR / 'battlesnake' / 'battlesnake.exe'

# Shared defaults for all steps. Absolute paths also work from another directory.
DATA_PATH = PROJECT_DIR / 'data' / 'cnn_global_moves.json'
MODEL_PATH = PROJECT_DIR / 'models' / 'cnn_global_model.pt'
RESULTS_PATH = PROJECT_DIR / 'evaluation' / 'cnn_global_survival.csv'
