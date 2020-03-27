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
    return stackframes([crop(grayscale(frame)) for frame in frames])

def random_action(episode, total_episodes):
    epsilon = 0
    fraction_of_total = episode/total_episodes

    if fraction_of_total <= 0.05:
        epsilon = 1

    if fraction_of_total>0.05 and fraction_of_total<=0.2:
        epsilon = 0.7

    if fraction_of_total>0.2 and fraction_of_total<=0.4:
        epsilon = 0.5

    if fraction_of_total>0.4 and fraction_of_total<=0.5:
        epsilon = 0.4

    if fraction_of_total>0.5 and fraction_of_total<=0.7:
        epsilon = 0.2

    if fraction_of_total>0.7:
        epsilon = 0.1

    return epsilon > np.random.random()


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