"""Step 3: apply the trained model to live Battlesnake /move requests."""
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from flask import Flask, request
from step_0_state_attributes import ACTIONS, INPUT_COLUMNS, state_to_attributes
from step_2_train_network import DirectionNetwork
from gamestate import GameState

class NeuralAgent:
    def __init__(self, model_path):
        torch.set_num_threads(1)
        saved = torch.load(model_path, map_location="cpu", weights_only=True)
        if saved["input_columns"] != INPUT_COLUMNS or saved["actions"] != ACTIONS:
            raise ValueError("Model columns/actions differ from this project. Record and train again after changing inputs.")
        self.columns = saved["input_columns"]
        self.mean = np.asarray(saved["mean"], dtype=np.float32)
        self.std = np.asarray(saved["std"], dtype=np.float32)
        # Use the same architecture students defined for training in Step 2.
        self.model = DirectionNetwork()
        self.model.load_state_dict(saved["weights"])
        self.model.eval()

    def move(self, state):
        # Exactly the same row and column order as the recorded DataFrame.
        row = pd.DataFrame([state_to_attributes(state)])
        inputs = row[self.columns].to_numpy(dtype=np.float32)
        inputs = torch.from_numpy((inputs - self.mean) / self.std)
        with torch.inference_mode():
            action_id = int(self.model(inputs).argmax(dim=1).item())
        return {"move": ACTIONS[action_id]}


def create_app(move, start=None, end=None, info=None):
    """Same four endpoints as the unchanged Exercise 1 server, without reloading."""
    app = Flask("Exercise 2 Battlesnake")

    @app.get("/")
    def on_info():
        return info() if info else {"apiversion":"1", "color":"#004a99", "head":"default", "tail":"default"}

    @app.post("/start")
    def on_start():
        if start: start(request.get_json())
        return "ok"

    @app.post("/move")
    def on_move():
        answer = move(GameState(request.get_json()))
        if not isinstance(answer, dict) or answer.get("move") not in ACTIONS:
            raise ValueError("Agent must return {'move': 'up'/'down'/'left'/'right'}")
        return answer

    @app.post("/end")
    def on_end():
        if end: end(request.get_json())
        return "ok"

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/rb_trained_nn_model.pt")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    agent = NeuralAgent(args.model)  # Load once; no training happens during play.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"Neural agent: http://127.0.0.1:{args.port}")
    create_app(agent.move).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
