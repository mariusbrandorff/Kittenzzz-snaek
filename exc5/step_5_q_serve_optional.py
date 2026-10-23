"""OPTIONAL: expose your trained policy to the regular Battlesnake engine.

Run this after Steps 1–4. No training happens in the HTTP callbacks. The table
was learned in a simplified solo environment; transfer to multiplayer is not
guaranteed. Set the CLI engine's board size to match settings.BOARD_SIZE.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import settings as cfg
from step_1_q_state import encode_state
from step_2_q_choose_action import choose_action
from supplied_experiment import load_table


def create_server(q, encoder, policy, port=8005):
    """Supplied API adapter, also usable for an automatic engine smoke test."""
    rng=np.random.default_rng(0)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,data):
            body=json.dumps(data).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(dict(apiversion='1',author='Game AI in Practice',color='#358e9d',head='default',tail='default'))
        def do_POST(self):
            observation=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/move':
                action=policy(q[encoder(observation)],0.,rng)
                self.reply(dict(move=cfg.ACTIONS[action]))
            else:self.reply({})
        def log_message(self,*args):pass
    return HTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8005)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    q=load_table(cfg.MODELS/f'q_seed_{args.seed}.npz')
    server=create_server(q,encode_state,choose_action,args.port)
    print(f'Q-learning Battlesnake listening on http://127.0.0.1:{args.port} (Ctrl+C to stop).')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
