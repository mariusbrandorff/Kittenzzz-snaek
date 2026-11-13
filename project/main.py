from argparse import BooleanOptionalAction, Namespace, ArgumentParser, _SubParsersAction
import json

from flask import Flask, request

from gamestate import GameState
from iAgent import IAgent
from agent_based.agent import Agent
from rule_based.agent import RuleBasedAgent


def main_agent(ns: Namespace) -> IAgent | None:
    if (ns.mode == "run"):
        print("Run agent")
        return Agent()
    else:
        print("Train agent")
        return None # Just trained it

def main_rule(ns: Namespace) -> IAgent:
    print("Rule run!")
    return RuleBasedAgent()

def agent_args(parser: ArgumentParser):
    parser.set_defaults(func=main_agent)

    subparser = parser.add_subparsers(title="modes", required=True)

    # Run
    run_parser = subparser.add_parser("run")
    run_parser.set_defaults(mode="run")

    ## Defaults
    run_parser.add_argument("-p", "--port", type=int, default=8000)
    run_parser.add_argument("-d", "--dry", action=BooleanOptionalAction, default=False)

    # Train
    train_parser = subparser.add_parser("train")
    train_parser.set_defaults(mode="train")

    train_parser.add_argument("-s", "--steps", type=int, default=20000)


def rule_args(parser: ArgumentParser):
    parser.set_defaults(func=main_rule)

    parser.add_argument("-i", "--iterations", type=int, default=64)

    # Defaults
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-d", "--dry", action=BooleanOptionalAction, default=False)

def arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparser = parser.add_subparsers(required=True)
    
    agent_args(subparser.add_parser("agent"))
    rule_args(subparser.add_parser("rule"))
    
    return parser

def main(agent: IAgent, port: int, args: Namespace):
    print(agent, port)
    print("Running")
    if (args.dry):
        agent.dry()
    else:
        print("Webserver")
        app = Flask("Battlesnake")

        @app.get("/")
        def on_info():
            return json.dumps(agent.info().to_json())

        @app.post("/start")
        def on_start():
            game_state = GameState(request.get_json())
            agent.start(game_state)
            return "ok"

        @app.post("/move")
        def on_move():
            game_state = GameState(request.get_json())
            return {"move": agent.move(game_state)}

        @app.post("/end")
        def on_end():
            game_state = GameState(request.get_json())
            agent.end(game_state)
            return "ok"

        @app.after_request
        def identify_server(response):
            response.headers.set(
                "server", "battlesnake/github/starter-snake-python"
            )
            return response
    
        app.run("0.0.0.0", port)
    

if __name__ == "__main__":
    args = arg_parser().parse_args()
    agent = args.func(args) # Just believe it exists
    if agent == None:
        print("Training done!")
        exit(0)
    main(agent, args.port, args)
