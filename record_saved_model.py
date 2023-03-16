import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque

from src.dqn import DQN
from src.tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--model_path", type=str, help="Path to the saved weigths")
    parser.add_argument("--width", type=int, default=10, help="Width of Tetris board")
    parser.add_argument("--height", type=int, default=20, help="Height of Tetris board")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")

    args = parser.parse_args()
    return args

def prepare_batch(memory, batch_size, device):
    # Pick out of rollout related state actions and resulting rewards
    batch = sample(memory, batch_size)
    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
    next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)
    reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)

    return state_batch, reward_batch, next_state_batch, done_batch

def evaluate(opt):
    # Check for cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Making the environment
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('output.avi', fourcc, 60.0, (int(1.5 * env.width*env.block_size), env.height*env.block_size))


    # Intialize the model, optimizer, and loss function
    model = DQN().to(device)
    state = env.reset().to(device)
    model.eval()

    # The memory to train the DQN
    done = False
    while not done:
        # Get state action pairs
        state_action_pairs = env.get_next_states()
        # Getting Actions and States
        actions, next_states = zip(*state_action_pairs.items())
        next_states = torch.stack(next_states).to(device)

        # Evaluating Network
        with torch.no_grad():
            Q_pred = model(next_states).squeeze()

        # Greedy Policy
        next_state = next_states[torch.argmax(Q_pred).item(), :]
        action = actions[torch.argmax(Q_pred).item()]
        
        # Obtain next state rewards and done
        reward, done = env.step(action, render=True, video=video)

        if done: # Gameover
            final_score = env.score
            final_tetrominoes = env.tetrominoes,
            final_cleared_lines = env.cleared_lines
            state = env.reset().to(device)
        else:
            state = next_state
            continue

        print("Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            final_score,
            final_tetrominoes,
            final_cleared_lines))

if __name__ == "__main__":
    opt = get_args()
    print(opt.model_path)
    checkpoint = torch.load(opt.model_path)
    print(checkpoint)
    # evaluate(opt)