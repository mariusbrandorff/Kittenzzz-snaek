"""Supplied board I/O, rendering and HTTP transport. No learner or search policy."""
from contextlib import closing
import json
import numpy as np

# HISSS order: up, right, down, left. All four moves are allowed, including collisions.
MOVES = ("up", "right", "down", "left")
DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))
TURN_LIMIT = 250


def seed_native(seed):
    from hisss.cpp.lib import CPP_LIB
    CPP_LIB.lib.set_seed(int(seed) % (2**31 - 1))  # Process-wide RNG in HISSS 1.3.0.


def new_game(seed):
    import hisss
    seed_native(seed)
    return hisss.BattleSnakeGame(hisss.BattleSnakeConfig(
        w=7, h=7, num_players=1, min_food=1, food_spawn_chance=0, all_actions_legal=True))


def board(game):
    """Use the same input schema in training and in an official /move request."""
    import hisss
    return json.loads(hisss.to_battlesnake_json(game, player=0))


def check_solo(state):
    b, rules = state["board"], state["game"].get("ruleset", {})
    if (len(b["snakes"]) != 1 or b.get("hazards") or
            rules.get("name", "solo") not in ("solo", "standard")):
        raise ValueError("Use a standard, hazard-free solo game.")


def from_battlesnake(state):
    """Start a new native search from this request, including stacked-tail length."""
    import hisss
    check_solo(state)
    b, s = state["board"], state["you"]
    rules = state["game"].get("ruleset", {}).get("settings", {})
    xy = lambda pts: [[p["x"], p["y"]] for p in pts]
    food = xy(b["food"])
    minimum = rules.get("minimumFood", 1)
    game = hisss.BattleSnakeGame(hisss.BattleSnakeConfig(
        w=b["width"], h=b["height"], num_players=1, all_actions_legal=True,
        init_turns_played=state["turn"], init_snake_pos={0:xy(s["body"])},
        init_snake_len=[s["length"]], init_snake_health=[s["health"]],
        init_food_pos=food, min_food=min(len(food), rules.get("minimumFood", 1)),
        food_spawn_chance=rules.get("foodSpawnChance", 15)))
    if minimum > len(food):
        # HISSS validates initial food count separately from the next-turn spawn rule.
        # Keep this exact board, then reinitialise with its actual future minimum.
        current = game.get_state()
        game.cfg.min_food = minimum
        game.set_state(current)
    return game


def advance(game, action):
    old_length = game.player_lengths()[0]
    _, done, _ = game.step((int(action),))
    apple = int(not done and game.player_lengths()[0] > old_length)
    return (-1.0 if done else 1.0 if apple else -0.01), done, apple


def rollout_moves(game):
    body = game.player_pos(0)
    x, y = body[0]
    occupied = body if game.player_lengths()[0] > len(body) else body[:-1]
    return [a for a, (dx, dy) in enumerate(DELTAS)
            if 0 <= x+dx < game.cfg.w and 0 <= y+dy < game.cfg.h
            and (x+dx, y+dy) not in occupied] or list(range(4))


def replay(policy, seed, gif, title):
    """Play one native game; policy(game) returns an action. Draw only for playback."""
    from PIL import Image, ImageDraw
    frames, apples, total = [], 0, 0.
    real_rng = np.random.default_rng(seed)
    with closing(new_game(seed)) as game:
        for turn in range(TURN_LIMIT + 1):
            state = board(game)
            live = bool(state["board"]["snakes"])
            im = Image.new("RGB", (308, 320), "#edf2f7")
            draw = ImageDraw.Draw(im)
            draw.text((12, 10), title, fill="#15243b")
            def tile(x,y):
                return (12+40*x, 30+40*(6-y), 50+40*x, 68+40*(6-y))
            for x in range(7):
                for y in range(7):
                    draw.rectangle(tile(x,y), fill="#15243b")
            for p in state["board"]["food"]:
                a,b,c,d = tile(p["x"],p["y"])
                draw.ellipse((a+8,b+8,c-8,d-8),fill="#ff6375")
            if live:
                s = state["you"]
                for p in s["body"]:
                    draw.rectangle(tile(p["x"],p["y"]),fill="#358e9d")
                draw.rectangle(tile(s["head"]["x"],s["head"]["y"]),fill="#ffe193")
            draw.text((12, 304), f'Turn {turn} | Apples {apples} | Return {total:.2f}'+
                      (' | eliminated' if not live else ''), fill="#15243b")
            frames.append(im)
            if not live or turn == TURN_LIMIT:
                break
            action = policy(game)
            # Simulations may consume native random draws: give real play its own seed stream.
            seed_native(real_rng.integers(2**31-1))
            reward, _, ate = advance(game, action)
            apples += ate
            total += reward
    gif.parent.mkdir(exist_ok=True, parents=True)
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=110, loop=0)
    print(f"Replay: {turn} turns, {apples} apples, return {total:.2f} -> {gif}")


def serve(move, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.reply(dict(apiversion="1", author="Game AI in Practice", color="#358e9d",
                            head="default", tail="default"))
        def do_POST(self):
            if self.path in ("/start", "/end"):
                self.reply({})
            elif self.path == "/move":
                try:
                    state = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    check_solo(state)
                    self.reply({"move": move(state)})
                except (ValueError, KeyError) as err:
                    self.reply({"error":str(err)}, 400)
            else:
                self.reply({}, 404)
        def log_message(self, *args):
            pass
    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Listening on http://127.0.0.1:{port}", flush=True)
        server.serve_forever()
