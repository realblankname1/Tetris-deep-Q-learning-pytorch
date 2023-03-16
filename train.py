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
    parser.add_argument("--width", type=int, default=10, help="Width of Tetris board")
    parser.add_argument("--height", type=int, default=20, help="Height of Tetris board")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of samples per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Rate")
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--memory_size", type=int, default=30000, help="Size of Rollout")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--score_saving", type=int, default=0, help="Saves Every new maximum score")
    parser.add_argument("--render", type=int, default=1, help="Render the Tetris board")
    parser.add_argument("--record", type=int, default=0, help="Records Video of Tetris. Only works if render is True")


    args = parser.parse_args()
    return args

def prepare_batch(memory, batch_size, cuda_available):
    # Pick out of rollout related state actions and resulting rewards
    batch = sample(memory, batch_size)
    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    device = torch.device('cuda' if cuda_available else 'cpu')

    state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
    next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)
    reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)

    return state_batch, reward_batch, next_state_batch, done_batch

def train(opt):
    # Linear Epsilon Decay Function
    def calc_epsilon(epoch):
        if opt.num_decay_epochs < epoch:
            return opt.final_epsilon
        return opt.final_epsilon + ((opt.num_decay_epochs - epoch) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

    # Check for cuda
    cuda_available = torch.cuda.is_available()

    # Making the environment
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    # Creating video object
    if opt.render and opt.record:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('output.avi', fourcc, 60.0, (int(1.5 * env.width*env.block_size), env.height*env.block_size))
    else:
        video=None

    # Intialize the model, optimizer, and loss function
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    state = env.reset()

    # Checking for CUDA
    if cuda_available:
        model.cuda()
        state = state.cuda()

    # The memory to train the DQN
    memory = deque(maxlen=opt.memory_size)
    epoch = 0
    epsilon = calc_epsilon(0)
    while epoch < opt.num_epochs:
        # Get state action pairs
        state_action_pairs = env.get_next_states()
        # Getting Actions and States
        actions, next_states = zip(*state_action_pairs.items())
        next_states = torch.stack(next_states)
        if cuda_available:
            next_states = next_states.cuda()
        # Evaluating Network
        model.eval()
        with torch.no_grad():
            Q_pred = model(next_states).squeeze()
        model.train()

        # e-Greedy Policy
        if epsilon >= random():
            ind = randint(0, len(actions) - 1)
        else:
            ind = torch.argmax(Q_pred).item()
        next_state = next_states[ind, :]
        action = actions[ind]
        
        # Obtain next state rewards and done
        reward, done = env.step(action, render=opt.render, video=video)
        # Append information to memory for later evaluation
        memory.append([state, reward, next_state, done])

        if done: # Gameover
            final_score = env.score
            final_tetrominoes = env.tetrominoes,
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if cuda_available:
                state = state.cuda()
        else:
            state = next_state
            continue
        # if the number of states in memory is smaller than batch size
        # repeat until memory is able to be sampled from
        if len(memory) < opt.memory_size / 10 or len(memory) < opt.batch_size:
            continue

        # Save Weights if new max score was reached
        if opt.score_saving:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

        # increment epoch 
        epoch += 1
        # Calculate New Epsilon
        epsilon = calc_epsilon(epoch)
        # Get Batch data
        state_batch, reward_batch, next_state_batch, done_batch = prepare_batch(memory, opt.batch_size, cuda_available)

        # Get associated q values
        q_sa = model(state_batch)
        # Obtain next states q values to compare
        model.eval()
        with torch.no_grad():
            q_sa_next = model(next_state_batch)
        model.train()
        # Calculate real future rewards with discount
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, q_sa_next)))[:, None]
        # Train the DQN
        optimizer.zero_grad()
        loss = criterion(q_sa, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))

if __name__ == "__main__":
    opt = get_args()
    train(opt)