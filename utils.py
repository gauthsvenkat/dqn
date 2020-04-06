from cv2 import resize
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque
import random

def grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def resize_frame(frame, size=(84,84)):
    return resize(frame[::2, ::2], dsize=size)

def stackframes(frames):
    assert len(frames) == 4, "Need 4 frames for stacking"
    return np.stack(frames, axis=-1)

def preprocess(frames):
    assert len(frames) == 4, "Need 4 frames for preprocessing"
    return stackframes([resize_frame(grayscale(frame)) for frame in frames])

def tensor(x, device=None):
    if len(x.shape) == 3:
        x = torch.tensor(x/255, dtype=torch.float).permute(2,0,1)
    elif len(x.shape) == 0:
        x = torch.tensor(x, dtype=torch.float)

    return x if device is None else x.to(device=device)


def process_batch(batch, target_model, num_actions, gamma, device):

    prev_state = torch.stack([tensor(i[0]) for i in batch])
    action = np.asarray([i[1] for i in batch])
    r = torch.stack([tensor(i[2]) for i in batch])
    next_state = torch.stack([tensor(i[3]) for i in batch])
    done = np.asarray([i[4] for i in batch])

    y = torch.zeros([len(batch), num_actions], dtype=torch.float)

    max_next_state_qvals = torch.max(target_model(next_state.to(device=device)).detach().cpu(), 1)[0]

    term_idx = np.where(done == True)[0]
    non_term_idx = np.where(done == False)[0]

    y[non_term_idx, action[non_term_idx]] = r[non_term_idx] + gamma * max_next_state_qvals[non_term_idx]
    y[term_idx, action[term_idx]] = r[term_idx]

    return prev_state, y

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, prev_state, action, reward, next_state, done):
        self.buffer.append([prev_state, action, reward, next_state, done])

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def length(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_actions=None, device=None):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)
        self.out = nn.Linear(512, num_actions)

        self.to(device)
        self.train() #always in train mode cause it only matters when we have batchnorm or dropout layers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc(x))
        x = self.out(x)
        
        return x