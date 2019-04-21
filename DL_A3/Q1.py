import torch
import torch.nn as nn
import torch.autograd as autograd
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
    assert (data[0][0].size()[1] == data[0][1].size()[1])
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
            critic.zero_grad()
            # Compute the gradient penalty
            a = torch.rand((batch_size,1))
            a = a.expand(p.size())
            z = a * p + (1 - a) * q
            z.requires_grad_()
            criticized_z = critic(z)
            gradients = autograd.grad(criticized_z, z, grad_outputs=torch.ones(criticized_z.size()), retain_graph=True, create_graph=True, only_inputs=True)[0]
            gradients = gradients.view(batch_size, -1)
#            print("--")
#            print(gradients)
#            print(gradients.size())
            print((gradients.norm(2, dim=1) - 1))
#            print((gradients.norm(2, dim=1) - 1).size())
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            # Compute the loss and optimize
            loss = -(torch.mean(critic(p)) - torch.mean(critic(q)) - lambda_ * gradient_penalty)
            loss.backward()
            optimizer.step()
    return critic
