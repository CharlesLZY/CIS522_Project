import argparse
import re

from Game import SnakeGame
from Agent import Agent
from BoringAgent import BoringAgent
from SearchAgent import GreedyAgent, AstarAgent
from QLearningAgent import QLearningAgent
from DeepQLearningAgent import DeepQLearningAgent
import torch
from AutoEncoder import Encoder

import utils

import imageio



parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width", type=int, default=64, help="window width")
parser.add_argument("-H", "--height", type=int, default=64, help="window height")
parser.add_argument("-b", "--block_size", type=int, default=20, help="block size")
parser.add_argument("-s", "--speed", type=int, default=50, help="refresh speed")
parser.add_argument("-v", "--verbose", action="store_true", help="print verbose")
parser.add_argument("-seed", type=int, default=0, help="random seed")
parser.add_argument("-a", "--agent", type=str, default="", choices=["", "random", "boring", "greedy", "astar", "qlearning", "deepq"], help="agent type")
parser.add_argument("-f", "--file", type=str, help="pretrained model")
parser.add_argument("-t", "--model_type", type=str, choices=["linear", "cnn", "encoder"], help="model type")
parser.add_argument("-e", "--encoder", type=str, help="pretrained encoder model")
parser.add_argument("-fs", "--feature_size", type=int, default=8, help="pretrained encoder model")
parser.add_argument("--eval_iters", type=int, default=10, help="number of iterations to evaluate model on")
parser.add_argument("--eval", action='store_true', help="whether to eval or show playouts")
parser.add_argument("--generate_gif", action='store_true', help="generate gif of evaluation rollout")


if __name__ == "__main__":

    args = parser.parse_args()

    ### create a game
    game = SnakeGame(W = args.width, 
                     H = args.height, 
                     BLOCK_SIZE=args.block_size, 
                     SPEED=args.speed, 
                     VERBOSE=args.verbose,
                     SEED=args.seed)
    
    ### set agent
    if not args.agent:
        game.SPEED = 5
    elif args.agent == "random":
        agent = Agent(game)
    elif args.agent == "boring":
        agent = BoringAgent(game)
    elif args.agent == "greedy":
        agent = GreedyAgent(game)
    elif args.agent == "astar":
        agent = AstarAgent(game)
    elif args.agent == "qlearning":
        if args.file:
            agent = QLearningAgent(game, pretrained_model=args.file)
        else:
            print("Please specify the model to load. Usage: main.py -a qlearning -f model.pkl")
            exit()
    elif args.agent == "deepq":
        if not args.file:
            print("Please specify the model to load. Usage: main.py -a qlearning -f model.pkl")
            exit()
        if not args.model_type:
            print("Please specify the model type. Usage: main.py -a qlearning -f model.pkl")
            exit()
        if args.model_type == "encoder":
            encoder = Encoder(W=args.width, H=args.height, feature_size=args.feature_size)
            encoder.encoder.load_state_dict(torch.load(args.encoder))
            agent = DeepQLearningAgent(game, "encoder", input_size=args.feature_size, encoder=encoder)
        else:
            agent = DeepQLearningAgent(game, model_type=args.model_type, pretrained_model=args.file, map_size=(args.height, args.width))
            
        
        agent.model.eval()
    
    if not args.eval:
        while True:
            game._play()
    else:
        # evaulation
        game.SPEED = 1000
        tot_reward = 0
        save_gif = 1
        max_iter = 400
        # perform eval_iters rollouts
        for i in range(args.eval_iters):
            game._restart()
            dead = 0
            frame = 0
            while not dead and frame < max_iter:
                reward, dead = game._play(save_gif=save_gif)
                frame += 1
                tot_reward += reward
                tot_reward *= 0.99
        print(f"Averaged reward over {args.eval_iters} evaluation iterations: {tot_reward/args.eval_iters}")

        if args.generate_gif:
            frame = 0
            for i in range(100):
                # dead = 0
                # while not dead and frame < max_iter:
                reward, dead = game._play(save_gif=save_gif, save_gif_frame=frame)
                frame += 1

                # if save_gif:
                #     last_frame = frame # save last frame for gif generation
                # save_gif = 0   # only want gif of one rollout
                tot_reward += reward

            # generate gif
            images = []
            filenames = [f"generate_gif/img{x}.png" for x in range(last_frame)]
            filenames = [f"generate_gif/img{x}.png" for x in range(100)]
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave('gif.gif', images, format="gif", fps=16)
                    