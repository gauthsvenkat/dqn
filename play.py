import argparse
import torch
import gym
from utils import preprocess, tensor, DQN
import time

parser = argparse.ArgumentParser(description='Play Pong')
parser.add_argument('--save_location', '-sl', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--episode', '-e', type=int)
parser.add_argument('--device', '-d', type=str, default=None)
args = parser.parse_args()

if not args.device:
    args.device = 'cpu'

env = gym.make('PongDeterministic-v4')

model = DQN(int(env.action_space.n), args.device)
model.load_state_dict(torch.load(args.save_location.format('model', args.episode)))

while True:
    observation = env.reset()
    buff = []
    done = False
    score = 0
    enemy_score = 0

    while not done:
        
        env.render()
        
        if len(buff) < 4:
            observation, reward, done, info = env.step(env.action_space.sample())
            buff.append(observation)
            continue

        x = tensor(preprocess(buff), args.device)[None]
        action = int(torch.argmax(model(x).detach().cpu()))

        observation, reward, done, info = env.step(action)

        buff.pop(0)
        buff.append(observation)
        
        if reward>0:
            score+=reward
        else:
            enemy_score-=reward

    print('Enemy Score - {}, Our Score - {}'.format(enemy_score, score))
    time.sleep(10)
