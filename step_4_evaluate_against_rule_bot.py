"""Step 4: compare neural and rule-based survival in separate solo games."""
import argparse
import json
import logging
from pathlib import Path
import random
import subprocess
import threading

import pandas as pd
from werkzeug.serving import make_server
import rule_based_agent as rule_based_agent
from step_3_neural_agent import NeuralAgent, create_app


def game_result(game, seed, starts, ends):
    """Compare two completed solo games using survival time and apples eaten."""
    if not all(name in starts and name in ends for name in ("Neural", "Rule-based")):
        raise RuntimeError("Missing agent start/end data; cannot score this game.")
    neural_steps = ends["Neural"]["turn"]
    rule_steps = ends["Rule-based"]["turn"]
    return {
        "Game": game,
        "Seed": seed,
        "Longer survival": ("Neural" if neural_steps > rule_steps else
                            "Rule-based" if rule_steps > neural_steps else "Equal"),
        "Points agent 1": ends["Neural"]["you"]["length"] - starts["Neural"]["you"]["length"],
        "Points agent 2": ends["Rule-based"]["you"]["length"] - starts["Rule-based"]["you"]["length"],
        "Neural time steps": neural_steps,
        "Rule-based time steps": rule_steps,
        "Difference (Neural - Rule)": neural_steps - rule_steps,
    }


def evaluate(model="models/rb_trained_nn_model.pt", games=10,
             engine="battlesnake/battlesnake.exe", seed_start=1000, seconds=120):
    if games < 1 or seconds <= 0:
        raise ValueError("Game count and time limit must be positive.")
    engine = Path(engine).resolve()
    if not engine.is_file():
        raise FileNotFoundError(f"Battlesnake simulator not found: {engine}")
    # Use different seeds from the recorded training games.
    metadata = Path(model).with_suffix(".json")
    if metadata.exists():
        used_seeds = json.loads(metadata.read_text(encoding="utf8"))["recorded_seeds"]
        if set(used_seeds) & set(range(seed_start, seed_start + games)):
            raise ValueError("Choose --seed-start outside the recorded training seeds.")

    # 1. Load the trained network and use our completed rule-based agent directly.
    neural = NeuralAgent(model)
    starts, ends, results = {}, {}, []

    def scored_app(name, move, start=None, end=None, info=None):
        def on_start(state):
            starts[name] = state
            if start:
                start(state)

        def on_end(state):
            ends[name] = state
            if end:
                end(state)

        return create_app(move, start=on_start, end=on_end, info=info)

    apps = [scored_app("Neural", neural.move),
            scored_app("Rule-based", rule_based_agent.move, start=rule_based_agent.start,
                       end=rule_based_agent.end, info=rule_based_agent.info)]

    # 2. Start one server thread for each agent. Port 0 chooses a free local port.
    servers, threads = [], []
    try:
        for app in apps:
            server = make_server("127.0.0.1", 0, app)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)

        # 3. Each seed is played twice: once alone by each agent.
        for game in range(games):
            starts.clear()
            ends.clear()
            seed = seed_start + game
            for name, server in zip(("Neural", "Rule-based"), servers):
                random.seed(seed)
                print(f"Pair {game + 1}/{games}, seed {seed}: {name} solo game", flush=True)
                command = [str(engine), "play", "-g", "solo", "-W", "11", "-H", "11",
                           "--seed", str(seed), "--timeout", "500",
                           "--name", name, "--url", f"http://127.0.0.1:{server.server_port}"]
                subprocess.run(command, check=True, timeout=seconds)
            results.append(game_result(game + 1, seed, starts, ends))
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()

    table = pd.DataFrame(results)
    print("\nAgent 1 = Neural | Agent 2 = Rule-based | Points = apples eaten")
    print(table.to_string(index=False))
    print(f"\nMean survival: Neural {table['Neural time steps'].mean():.1f} steps; "
          f"Rule-based {table['Rule-based time steps'].mean():.1f} steps")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/rb_trained_nn_model.pt")
    parser.add_argument("--games", type=int, default=10, help="Seed pairs; each agent plays one solo game per seed")
    parser.add_argument("--engine", default="battlesnake/battlesnake.exe")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    evaluate(args.model, args.games, args.engine, args.seed_start, args.seconds)
