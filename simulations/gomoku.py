#!/usr/bin/env python3
"""
Gomoku Bot Simulation Script

This script sets up a Gomoku board simulation where a Docker-based Gomoku bot
plays moves via SocketIO. The simulation supports a board state that can be updated
mid-game (e.g. spawning random blocks) and logs game history to a file.

Key Features:
    - Board represented as a 2D numpy array.
    - Two players: "black" and "white" (values 1 and 2 in the board, respectively).
    - Ability to spawn a random block (value 3) on the board after moves.
    - Communication with the Gomoku bot Docker container via SocketIO events.
    - Command-line arguments to configure board dimensions, winning condition,
      time limit per turn, and output log file.
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from copy import deepcopy

import docker
import numpy as np
import socketio


@dataclass
class GameEvent:
    """
    Represents any game event we want to track in history.
    """

    row: int
    col: int
    stone: int


@dataclass
class GameState:
    """
    Represents the current state of the Gomoku game.

    Attributes:
        board (np.ndarray): 2D array (HxW) representing the board.
            Board cell values:
                0 -> empty,
                1 -> black stone,
                2 -> white stone,
                3 -> block.
        current_move (str): Indicates which player's turn is next ("black" or "white").
        total_steps (int): Total number of moves made so far.
        history (list): history of all game events.
    """

    board: np.ndarray
    current_move: str
    total_steps: int
    history: list

    def __init__(self, h: int, w: int):
        self.board = np.zeros((h, w), dtype=int)
        self.current_move = "black"
        self.total_steps = 0
        self.history = []

    def next_step(self, event: GameEvent) -> None:
        """Advances the move counter and alternates the current player."""
        if event.stone in [1, 2]:
            # A bot made a move
            self.total_steps += 1
            self.current_move = "white" if self.current_move == "black" else "black"
        self.board[event.row, event.col] = event.stone
        self.history.append(event)

    def __repr__(self):
        return self.history.__repr__()


def pos_to_coords(pos: int, h: int, w: int) -> tuple[int, int]:
    """
    Converts a board position (provided as an index in row-major order)
    into (row, col) coordinates.

    Args:
        pos (int): The position index.
        h (int): Board height.
        w (int): Board width.

    Returns:
        tuple[int, int]: (row, col) coordinates.
    """
    row = pos // w
    col = pos % w
    return row, col


def board_visual(board: np.ndarray) -> str:
    """
    Generates a string representation of the game board.

    Symbols used:
        "." for an empty cell,
        "X" for a black stone,
        "O" for a white stone,
        "*" for a block.

    Args:
        board (np.ndarray): The game board as a 2D numpy array.

    Returns:
        str: A multi-line string visualizing the board with row and column headers.
    """
    h, w = board.shape
    symbols = {0: ".", 1: "X", 2: "O", 3: "*"}
    # Column headers labeled with letters starting at A.
    header = "   " + " ".join(chr(ord("A") + i) for i in range(w))
    lines = [header]
    for r in range(h):
        # Row header using letters (Note: For large boards, labeling may run past 'Z'.)
        row_label = chr(ord("A") + r)
        row_str = " ".join(symbols.get(cell, "?") for cell in board[r])
        lines.append(f"{row_label}  {row_str}")
    return "\n".join(lines)


def print_board_state(game_state: GameState) -> None:
    """
    Prints the current board state along with move information.

    Args:
        game_state (GameState): The current state of the game.
    """
    print(board_visual(game_state.board))
    print(f"Total moves so far: {game_state.total_steps}")
    print(f"Next move: {game_state.current_move}\n")


def start_docker_container(
    args,
    image: str = "gomokuai:latest",
    host_port: int = 7682,
) -> docker.models.containers.Container:
    """
    Starts a Docker container running the Gomoku bot.

    Args:
        image (str): Name of the Docker image (default: "gomokuai:latest").
        host_port (int): Host port to map the container's port to (default: 7682).

    Returns:
        docker.models.containers.Container: The running container instance.

    Exits the script if container startup fails.
    """
    client = docker.from_env()
    try:
        if args.v:
            print("Starting Gomoku bot container...")
        container = client.containers.run(
            image, ports={"7682/tcp": host_port}, detach=True
        )
    except Exception as ex:
        print("Error starting Docker container:", ex)
        sys.exit(1)
    # Allow time for the container to initialize.
    time.sleep(3)
    return container


def parse_parameters() -> argparse.Namespace:
    """
    Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including board dimensions,
        number of stones to win, time limit per turn, and output log file.
    """
    parser = argparse.ArgumentParser(
        description="Simulate a Gomoku game between two bots."
    )
    parser.add_argument(
        "-W", type=int, default=15, help="Width of the board (default: 15)"
    )
    parser.add_argument(
        "-H",
        type=int,
        default=0,
        help="Height of the board. If 0, the board is square (default: 0)",
    )
    parser.add_argument(
        "-n", type=int, default=5, help="Number of stones in a row to win (default: 5)"
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=0.5,
        help="Time limit per turn for the bot in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--generate_cf",
        action="store_true",
        help="If used, generates two runs that are splitted after -b steps.",
    )
    parser.add_argument(
        "-b",
        type=int,
        default=15,
        help="Branching point. The number of steps to generate the second wave of random blocks. The universe is splitted into two if --generate_cf used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="game_history.txt",
        help="Path to the game history log file (default: game_history.txt)",
    )
    parser.add_argument(
        "--init_blocks",
        type=int,
        default=15,
        help="Block to spawn at the map initiation.",
    )
    parser.add_argument(
        "--add_blocks", type=int, default=15, help="Block to spawn during the run."
    )
    parser.add_argument("-v", action="store_true", help="Print states in stdout")
    args = parser.parse_args()
    if args.H == 0:
        args.H = args.W  # Make the board square if height is unspecified.
    return args


def board_to_cmd(game_state: GameState, args: argparse.Namespace) -> list[str]:
    """
    Generates a list of command-line arguments representing the current game state.

    This command is used to instruct the Gomoku bot process on the current board
    and simulation parameters. It includes board dimensions, game parameters,
    the current player, and positions of stones and blocks.

    Args:
        game_state (GameState): Current game state.
        args (argparse.Namespace): Command-line parameters.

    Returns:
        list[str]: A list of command-line arguments to pass to the bot.
    """
    board = game_state.board
    h, w = board.shape

    # Basic parameters
    cmd = [
        "-w",
        str(w),
        "-H",
        str(h),
        "-n",
        str(args.n),
        "-tl",
        str(args.time_limit),
        "-vq",  # very quiet mode for reduced verbosity.
    ]

    # Configure players: one AI vs one human
    # It makes the server make one move and wait
    if game_state.current_move == "white":
        cmd.extend(["-hm1", "-ai2"])
        cmd.append("-ws")
    else:
        cmd.extend(["-ai1", "-hm2"])

    # Encode the current board state by listing occupied positions:
    # Positions are given as a sequential index, row-major order.
    black = []
    white = []
    block = []
    for i in range(h):
        for j in range(w):
            pos = str(i * w + j)
            if board[i, j] == 1:
                black.append(pos)
            elif board[i, j] == 2:
                white.append(pos)
            elif board[i, j] == 3:
                block.append(pos)

    if black:
        cmd.append("--black")
        cmd.extend(black)
    if white:
        cmd.append("--white")
        cmd.extend(white)
    if block:
        cmd.append("--block")
        cmd.extend(block)

    cmd.append("--json-client")
    return cmd


def spawn_one_random_block(game_state: GameState, prob: float = 1.0) -> int | None:
    """
    Randomly places a block (value 3) on the board if a free cell is available.

    With the specified probability, the function selects one of the empty cells
    and updates the board state by placing a block there. The board cell index
    is returned if a block is placed, otherwise None is returned.

    Args:
        game_state (GameState): The current game state.
        prob (float): Probability to place a block (default: 1.0).

    Returns:
        int | None: The board index where the block was placed, or None if not placed.
    """
    h, w = game_state.board.shape
    free_positions = [
        (i, j) for i in range(h) for j in range(w) if game_state.board[i, j] == 0
    ]
    if not free_positions:
        return None
    row, col = random.choice(free_positions)
    if random.uniform(0, 1) < prob:
        game_state.next_step(GameEvent(row, col, stone=3))  # place a block
        return row * w + col
    return None


def spawn_random_blocks(game_state, n=10, prob=1.0):
    for _ in range(n):
        spawn_one_random_block(game_state)


def main() -> None:
    args = parse_parameters()
    game_state = GameState(args.H, args.W)

    # Open the game history file using a context manager for safe resource handling.
    container = start_docker_container(args)
    sio = socketio.Client()

    # Global flags for simulation control.
    global is_simulation_over, is_server_ready
    is_simulation_over = False  # True when game-over condition is detected.
    is_server_ready = True  # Set to True once the server completes its move.

    @sio.event
    def connect() -> None:
        """Triggered upon successful SocketIO connection."""
        if args.v:
            print("Connected to Gomoku server.")

    @sio.event
    def disconnect() -> None:
        """Triggered when the SocketIO client disconnects."""
        if args.v:
            print("Disconnected from Gomoku server.")

    @sio.on("place stone")
    def on_place_stone(data: dict) -> None:
        """
        Handles the placement of a stone from the server.

        Updates the board state, logs the move, and toggles the current player.

        Args:
            data (dict): Data from the server containing:
                - "pos": Position index on the board.
                - "stone": The stone type (1, 2, etc.).
        """
        global is_server_ready, is_simulation_over
        if is_simulation_over:
            return

        pos = data.get("pos")
        stone = data.get("stone")
        if pos is None:
            return

        row, col = pos_to_coords(pos, args.H, args.W)

        if game_state.board[row, col] == stone:
            # When you start the game with a non-empty board,
            # it simulates the previous steps calling on_place_stone()
            # We ignore that, as our board is already have these steps.
            return
        if game_state.board[row, col] != 0:
            # That's strange and bad
            print("!!! Desynchronization with the server detected !!!")

        # Update game state.
        game_state.next_step(GameEvent(row, col, stone))

        if args.v:
            move_msg = (
                f"Bot placed stone at {chr(ord('A') + row)}{chr(ord('A') + col)} "
                f"(server index {pos}), stone: {stone}"
            )
            print(move_msg)
            board_str = board_visual(game_state.board)
            print(board_str)
        is_server_ready = True

    @sio.on("log")
    def on_log(data: dict) -> None:
        """
        Processes log messages received from the server.
        It checks for game-over conditions by scanning for specific keywords.

        Args:
            data (dict): Contains a "msg" key with log information.
        """
        global is_simulation_over
        if is_simulation_over:
            return
        msg = data.get("msg", "")
        log_msg = "Log: " + msg
        if args.v:
            print(log_msg)
        # Check for signals of game termination.
        lowered = msg.lower()
        if any(x in lowered for x in ("stop", "finished", "won")):
            if args.v:
                print("Game over detected from log event.")
            if "won" in lowered:
                winner = lowered.split()[6]  # the winner player
                game_state.history.append(winner)
            elif "draw" in lowered:
                game_state.history.append("draw")
            else:
                print("Unknown message:", lowered)
            is_simulation_over = True

    @sio.on("log error")
    def on_log_error(data: dict) -> None:
        """
        Handles error log messages from the server.

        Args:
            data (dict): Contains a "msg" key detailing the error.
        """
        msg = data.get("msg", "")
        err_msg = "Error log: " + msg
        print(err_msg)

    try:
        # Try to establish a connection to the Gomoku server.
        sio.connect("http://localhost:7682")
    except Exception as e:
        print("Socket connection error:", e)
        container.stop()
        sys.exit(1)

    # Main simulation loop:
    # The Gomoku bot supports human vs bot and two bots scenarios.
    # To allow mid-game board state modifications, we use the following trick:
    #   1. The server places a stone.
    #   2. We update our GameState accordingly.
    #   3. We emit a "new game" event with the updated state.
    # This loop continues until a game-over condition is detected.
    spawn_random_blocks(game_state, n=args.init_blocks)
    if args.v:
        print("Initial board:")
        print(board_visual(game_state.board))

    runs = [game_state]
    try:
        while not is_simulation_over:
            is_server_ready = False
            cmd = board_to_cmd(game_state, args)
            sio.emit("new game", {"args": cmd})

            # Wait until the server has processed and made a move.
            while not is_server_ready:
                time.sleep(0.1)

            if game_state.total_steps == args.b:
                if args.generate_cf:
                    alternative_game_state = deepcopy(game_state)
                    runs.append(
                        alternative_game_state
                    )  # save the alternative run to simulate it later
                spawn_random_blocks(game_state, n=args.add_blocks)
                if args.v:
                    print("Intermediate board state:")
                    print(board_visual(game_state.board))

        if args.generate_cf:
            if args.v:
                print(
                    "First universe is over. Simulation of the alternative universe..."
                )
            if len(runs) == 1:
                if args.v:
                    print(
                        "The simulation is over before the branching point. Finishing the run..."
                    )
            else:
                game_state = runs[-1]
                spawn_random_blocks(game_state, n=args.add_blocks)
                is_simulation_over = False

                while not is_simulation_over:
                    is_server_ready = False
                    cmd = board_to_cmd(game_state, args)
                    sio.emit("new game", {"args": cmd})

                    # Wait until the server has processed and made a move.
                    while not is_server_ready:
                        time.sleep(0.1)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Cleanup: disconnect from the server and stop the container.
        sio.disconnect()
        container.stop()
        with open(args.output, "w") as f:
            f.write(runs.__repr__())
        if args.v:
            print(
                f"Container stopped. Simulation ended. Game history saved to {args.output}"
            )


if __name__ == "__main__":
    main()
