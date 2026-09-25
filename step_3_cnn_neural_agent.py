"""Step 3: use the trained CNN for live Battlesnake /move requests."""
import argparse
import logging
import torch
import cnn_config as config
from step_0_cnn_state_attributes import ACTIONS, CHANNELS, INPUT_SHAPE, ENCODING_VERSION, cnn_state_to_attributes
from step_2_cnn_train_network import DirectionNetwork
from server import create_app as starter_app


class CNNAgent:
    def __init__(self, model_path=None):
        torch.set_num_threads(1)
        saved = torch.load(model_path if model_path is not None else config.MODEL_PATH, map_location='cpu', weights_only=True)
        if (saved.get('channels') != CHANNELS or tuple(saved.get('input_shape', ())) != INPUT_SHAPE
                or saved.get('actions') != ACTIONS or saved.get('encoding_version') != ENCODING_VERSION):
            raise ValueError('Checkpoint and Step 0 encoding differ. Re-encode the data and retrain the global CNN.')
        self.recorded_seeds = saved['recorded_seeds']
        self.model = DirectionNetwork()
        self.model.load_state_dict(saved['weights'])
        self.model.eval()

    def cnn_move(self, state):
        inputs = cnn_state_to_attributes(state).unsqueeze(0)  # (1,C,H,W), using the recording encoder.
        with torch.inference_mode():
            direction = ACTIONS[int(self.model(inputs).argmax(1).item())]
        # No rule-based correction: evaluate what the network actually learned.
        return {'move': direction}


def cnn_create_app(move, start=None, end=None, info=None):
    return starter_app({
        'move': move, 'start': start or (lambda state: None), 'end': end or (lambda state: None),
        'info': info or (lambda: {'apiversion': '1', 'color': '#16a085', 'head': 'default', 'tail': 'default'}),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_PATH)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    agent = CNNAgent(args.model)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(f'CNN agent: http://127.0.0.1:{args.port}')
    cnn_create_app(agent.cnn_move).run(host='127.0.0.1', port=args.port, debug=False)
