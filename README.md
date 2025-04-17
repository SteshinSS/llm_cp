# LLM Counterfactual Prediction
This repository studies the ability of LLMs to make counterfactual predictions.

## Installation
To install basic tools, run:
```
conda create --name llm_cp python=3.11
conda activate llm_cp
pip install -r requirements.txt
```

To use models API you need to create `.env` file with keys:
```
OPENAI_API_KEY=sk-proj....
```

Each simulation may require additional setup, as indicated below.


## Simulations
1) Gomoku

Gomoku is a generalization of Tic-Tac-Toe to an H×W grid, where the goal is to place N stones in a row. Additionally, the board can include blocked cells as obstacles.
To run Gomoku simulations, install the Gomoku bot and use `simulations/gomoku.py`:
```
usage: gomoku.py [-h] [--w W] [--h H] [--n N] [--time_limit TIME_LIMIT] [--output OUTPUT] [--v]

Simulate a Gomoku game between two bots.

options:
  -h, --help            show this help message and exit
  --w W                 Width of the board (default: 15)
  --h H                 Height of the board. If 0, the board is square (default: 0)
  --n N                 Number of stones in a row to win (default: 5)
  --time_limit TIME_LIMIT
                        Time limit per turn for the bot in seconds (default: 0.5)
  --output OUTPUT       Path to the game history log file (default: game_history.txt)
  --v                   Print states in stdout
```

### Installation
The Gomoku simulator uses a Docker container. To build it, clone the [Gomoku-AI repository](https://github.com/rdragon/gomoku-ai/tree/main) and build the Docker container:
```
docker build --pull --rm -f "Dockerfile" -t gomokuai:latest .
```


## Development
We use ruff for code formatting.

We use pip-tools to pin dependencies. If you need to add a new package, just add it to `requirements.in` and run:
```
conda activate llm_cp
pip-compile
pip-sync
```
That will update `requirements.txt` and update your current environment.

We keep the API keys in `.env` file, which is in `.gitignore` and should never be commited to the repo. To use the keys in code:
```Python
from dotenv import load_dotenv
load_dotenv()
```