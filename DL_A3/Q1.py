import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

def q1(data):
    assert (data[0][0].size()[1] == data[0][1].size()[1])
    input_size = data[0][0].size()[1]
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )
    optimizer = optim.SGD(discriminator.parameters(), lr=0.001)
    for epoch in range(10):
        for (p, q) in data:
            discriminator.zero_grad()
            loss = math.log(2) + torch.mean(torch.log(discriminator(p))) / 2 + torch.mean(torch.log(1 - discriminator(q))) / 2
            loss.backward()
            optimizer.step()
    return discriminator

def q2(data):
    assert (data[0].size()[1] == data[1].size()[1])
    batch_size = data[0][0].size()[0]
    input_size = data[0][0].size()[1]
    lambda_ = 10
    critic = torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    optimizer = optim.SGD(critic.parameters(), lr=0.001)
    for epoch in range(10):
        for (p, q) in data:
            discriminator.zero_grad()
            # TODO add gradient penalty, see https://kionkim.github.io/2018/07/26/WGAN_3/ get_penalty
            a = torch.rand((batch_size,1))
            z = critic(a * p + (1 - a) * q)
            loss = -(torch.mean(critic(p)) - torch.mean(critic(q)) - lambda_ * grad )
            loss.backward()
            optimizer.step()
    return critic