import torch
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from time import time
import gym
import cv2
import numpy as np
from utils import ConvNet, preprocess, process_batch
from torchvision.transforms import ToTensor
import os
import random
from collections import deque

parser = argparse.ArgumentParser(description='Train DQN')
parser.add_argument('--save_location', '-sl', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--episodes', '-e', type=int, default=1000)
parser.add_argument('--save_every', '-se', type=int, default=100)
parser.add_argument('--device', '-d', type=str, default=None)
parser.add_argument('--replay_size', '-rs', type=int, default=1000)
parser.add_argument('--render_env', '-re', action='store_true')
parser.add_argument('--batch_size', '-bs', type=int, default=64)
args = parser.parse_args()

if not args.device:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('BreakoutNoFrameskip-v4')

LEARNING_RATE = 1e-4
GAMMA = 0.9
REPLAY_MEMORY = []
EPSILON = 1

print('Loading model')
tensor = ToTensor()
model = ConvNet(num_out=int(env.action_space.n), device=args.device)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#initialize replay memory to capacity 1000

print('Initializing replay memory with {} samples'.format(args.replay_size))

while len(REPLAY_MEMORY) < args.replay_size:
    observation = env.reset()
    buff = deque()
    prev_buff = []
    done = False

    while not done:

        prev_buff = buff
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if len(buff) < 4:
            buff.appendleft(observation)
            continue

        buff.pop(); buff.appendleft(observation)

        previous_state = preprocess(prev_buff)    

        r = -1 if done else reward
        next_state = preprocess(buff)

        REPLAY_MEMORY.append((previous_state, action, r, next_state))

REPLAY_MEMORY = REPLAY_MEMORY[-args.replay_size:] #trim to have only the last number of memories

for e in range(args.episodes):
    observation = env.reset()
    buff = deque()
    prev_buff = []
    done = False

    episode_loss = 0.0
    episode_steps = 0
    episode_reward = 0

    print('Episode {} - '.format(e+1), end='')

    while not done:

        if args.render_env:
            env.render()

        prev_buff = buff

        if len(buff) < 4:
            observation, reward, done, info = env.step(env.action_space.sample())
            buff.appendleft(observation)
            continue

        previous_state = preprocess(prev_buff)
        x = previous_state[None].to(device=args.device)

        action = env.action_space.sample() if EPSILON > random.random() else int(torch.argmax(model(x)))
        #print(' Action - ', action)
        observation, reward, done, info = env.step(action)

        buff.pop(); buff.appendleft(observation)

        r = -1 if done else reward
        next_state = preprocess(buff)

        REPLAY_MEMORY.pop(0); REPLAY_MEMORY.append((previous_state, action, r, next_state))

        #DO THE ACTUAL LEARNING

        prev_states, ys = process_batch(random.sample(REPLAY_MEMORY, args.batch_size), model, args.device, int(env.action_space.n), GAMMA)

        optimizer.zero_grad()
        loss = mse_loss(model(prev_states.to(device=args.device)), ys.to(device=args.device))
        loss.backward()
        optimizer.step()

        episode_loss+=loss.item()
        episode_steps+=1
        episode_reward+=reward

    if EPSILON > 0.01:
        EPSILON-=(1.0/(e+1))

    print('Total steps = {}, Reward = {} , Average Loss = {}'.format(episode_steps, episode_reward, episode_loss/episode_steps))

    if (e+1)%args.save_every == 0:
        if not os.path.exists('model/'):
            os.mkdir('model/')

        torch.save(model.state_dict(), args.save_location.format('model', e+1))
        torch.save(optimizer.state_dict(), args.save_location.format('opt', e+1))
        print('Save model and optimizer weights for episode {}'.format(e+1))

env.close()











    






