import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import gym
from utils import DQN, preprocess, process_batch, ReplayBuffer, tensor
import os
from collections import deque
import random
import numpy as np

parser = argparse.ArgumentParser(description='Train DQN')
parser.add_argument('--save_location', '-sl', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--episodes', '-e', type=int, default=1000)
parser.add_argument('--save_every', '-se', type=int, default=100)
parser.add_argument('--device', '-d', type=str, default=None)
parser.add_argument('--replay_size', '-rs', type=int, default=10000)
parser.add_argument('--render_env', '-re', action='store_true')
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--start_training', '-st', type=int)
parser.add_argument('--update_net_every', '-un', type=int, default=100)
parser.add_argument('--epsilon', '-ep', type=float, default=1)
parser.add_argument('--eps_decay', '-epd', type=float, default=0.01)
parser.add_argument('--min_epsilon', '-mep', type=float, default=0.01)
parser.add_argument('--gamma', '-g', type=float, default=0.99)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.00025)
args = parser.parse_args()

if not args.device:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.start_training = args.replay_size//2 if not args.start_training else args.start_training

env = gym.make('PongDeterministic-v4')

REPLAY_MEMORY = ReplayBuffer(args.replay_size, args.batch_size)

print('Loading model')
model = DQN(int(env.action_space.n), args.device)
target_model = DQN(int(env.action_space.n), args.device)
target_model.load_state_dict(model.state_dict())
huber_loss = nn.SmoothL1Loss()
optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.95, eps=0.01)

#initialize replay memory

print('Initializing replay memory with {} samples'.format(args.start_training))

while REPLAY_MEMORY.length() < args.start_training:
    observation = env.reset()
    buff = deque(maxlen=4)
    prev_buff = []
    done = False

    while not done:

        prev_buff = buff
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if len(buff) < 4:
            buff.append(observation)
            continue

        buff.append(observation)

        previous_state = preprocess(prev_buff)    
        next_state = preprocess(buff)

        REPLAY_MEMORY.put(previous_state, np.uint8(action), np.int8(reward), next_state, done)


for e in range(args.episodes):
    observation = env.reset()
    buff = deque(maxlen=4)
    prev_buff = []
    done = False

    episode_loss = 0.0
    episode_steps = 0
    episode_reward = 0

    print('Episode {} - Epsilon = {}, '.format(e+1, args.epsilon), end='')

    while not done:

        if args.render_env:
            env.render()

        prev_buff = buff

        if len(buff) < 4:
            observation, reward, done, info = env.step(env.action_space.sample())
            buff.append(observation)
            continue

        previous_state = preprocess(prev_buff)

        if args.epsilon > random.random():
            action = env.action_space.sample()
        else:
            x = tensor(previous_state, args.device)[None]
            action = np.uint8(torch.argmax(model(x).detach().cpu()))

        observation, reward, done, info = env.step(action)

        buff.append(observation)

        next_state = preprocess(buff)

        REPLAY_MEMORY.put(previous_state, action, np.int8(reward), next_state, done)

        #DO THE ACTUAL LEARNING

        prev_states, ys = process_batch(REPLAY_MEMORY.sample(), target_model, int(env.action_space.n), args.gamma, args.device)

        optimizer.zero_grad()
        loss = huber_loss(model(prev_states.to(device=args.device)), ys.to(device=args.device))
        loss.backward()
        optimizer.step()

        episode_loss+=loss.item()
        episode_steps+=1
        episode_reward+=reward

    if args.epsilon > args.min_epsilon:
        args.epsilon-=args.epsilon*args.eps_decay

    if (e+1)%args.update_net_every == 0:
        target_model.load_state_dict(model.state_dict())
        #print(' Updated target q network weights ')

    print('Total steps = {}, Reward = {} , Average Loss = {}'.format(episode_steps, episode_reward, episode_loss/episode_steps))

    if (e+1)%args.save_every == 0:
        if not os.path.exists('model/'):
            os.mkdir('model/')

        torch.save(model.state_dict(), args.save_location.format('model', e+1))
        torch.save(optimizer.state_dict(), args.save_location.format('opt', e+1))
        print('Saved model and optimizer weights for episode {}'.format(e+1))

env.close()











    






