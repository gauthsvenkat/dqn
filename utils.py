import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#Dictionary holding the crop details for the incoming frame (these values are for breakout)
CROP = {'TOP':33,
        'BOTTOM':195,
        'LEFT':8,
        'RIGHT':152}


def crop(frame, crop_details=CROP):
    assert len(frame.shape) in [2,3]
    return frame[CROP['TOP']:CROP['BOTTOM'], CROP['LEFT']:CROP['RIGHT'], :] if len(frame.shape) == 3 else frame[CROP['TOP']:CROP['BOTTOM'], CROP['LEFT']:CROP['RIGHT']]

def grayscale(frame):
    assert len(frame.shape) == 3, "Input does not have 3 dimensions!"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def stackframes(frames):
    assert len(frames) == 4, "Need 4 frames for stacking"
    return np.stack(frames, axis=-1)

def preprocess(frames):
    assert len(frames) == 4, "Need 4 frames for preprocessing"
    p = stackframes([crop(grayscale(frame)/255) for frame in frames])

    return torch.tensor(p, dtype=torch.float).permute(2,0,1)

def process_batch(batch, model, device, num_actions, gamma):

    prev_state = torch.stack([i[0] for i in batch])
    action = np.asarray([i[1] for i in batch])
    r = torch.stack([torch.tensor(i[2], dtype=torch.float) for i in batch])
    next_state = torch.stack([i[3] for i in batch])

    y = torch.zeros([len(batch), num_actions], dtype=torch.float)

    max_next_state_qvals = torch.max(model(next_state.to(device=device)).detach().cpu(), 1)[0]

    term_idx = np.where(r == -10)[0]
    non_term_idx = np.where(r != -10)[0]

    y[non_term_idx, action[non_term_idx]] = r[non_term_idx] + gamma * max_next_state_qvals[non_term_idx]
    y[term_idx, action[term_idx]] = r[term_idx]

    return prev_state, y


class ConvNet(nn.Module):
    def __init__(self, num_out=None, device=None):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.fc = nn.Linear(4608, 128)
        self.out = nn.Linear(128, num_out)

        self.to(device)
        self.train() #always in train mode cause it only matters when we have batchnorm or dropout layers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.out(x)
        
        return x