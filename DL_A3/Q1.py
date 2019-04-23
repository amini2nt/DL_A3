#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.random as trandom
import numpy as np
import math
import random
import matplotlib.pyplot as plt


############### HELPERS ##############
def show_ws():
    pairs = [(l.split("\t")[0], l.split("\t")[1]) for l in open("ws.tab").read().split("\n") if l]
    x = [float(k[0]) for k in pairs]
    y = [float(k[1]) for k in pairs]
    plt.scatter(x, y)
    plt.show()

def show_js():
    pairs = [(l.split("\t")[0], l.split("\t")[1]) for l in open("js.tab").read().split("\n") if l]
    x = [float(k[0]) for k in pairs]
    y = [float(k[1]) for k in pairs]
    plt.scatter(x, y)
    plt.show()

def distribution1(x, batch_size=1):
    # Distribution defined as (x, U(0,1)). Can be used for question 3
    while True:
        a = np.array([(x, random.uniform(0, 1)) for _ in range(batch_size)] , dtype=np.double)
        yield(torch.from_numpy(a))

def distribution3(batch_size=1):
    # High dimension gaussian distribution
    while True:
        yield(torch.from_numpy(np.random.normal(0, 1, (batch_size, 1)))) # Can be used in 1.4

e = lambda x: np.exp(x)
tanh = lambda x: (e(x) - e(-x)) / (e(x)+e(-x))
def distribution4(batch_size=1):
    # arbitrary sampler
    f = lambda x: tanh(x*2+1) + x*0.75
    while True:
        yield(torch.from_numpy(f(np.random.normal(0, 1, (batch_size, 1)))))


############## QUESTION 1 ###############
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
    discriminator.double()
    optimizer = optim.SGD(discriminator.parameters(), lr=0.01)
    for epoch in range(10):
        #print("JS Epoch #%s" % (epoch+1))
        for (p, q) in data:
            discriminator.zero_grad()
            loss = -(torch.mean(torch.log(discriminator(p))) / 2 + torch.mean(torch.log(1 - discriminator(q))) / 2)
            loss.backward()
            optimizer.step()
        #print("JS Loss :%s" % loss.data.item())
    def js_distance(p,q):
        return (math.log(2) + torch.mean(torch.log(discriminator(p))) / 2 + torch.mean(torch.log(1 - discriminator(q))) / 2).item()
    return js_distance



############## QUESTION 2 ###############
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
    critic.double()
    optimizer = optim.SGD(critic.parameters(), lr=0.001)
    for epoch in range(10):
        #print("WS Epoch #%s" % (epoch+1))
        for (p, q) in data:
            critic.zero_grad()
            # Compute the gradient penalty
            a = torch.rand((batch_size,1)).double()
            a = a.expand(p.size())
            z = a * p + (1 - a) * q
            z.requires_grad_()
            criticized_z = critic(z)
            outputs = torch.ones(criticized_z.size()).double()
            gradients = autograd.grad(criticized_z, z, grad_outputs=outputs, retain_graph=True, create_graph=True, only_inputs=True)[0]
            gradients = gradients.view(batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            # Compute the loss and optimize
            loss = -(torch.mean(critic(p)) - torch.mean(critic(q)) - lambda_ * gradient_penalty)
            loss.backward()
            optimizer.step()
        print("WS Loss :%s" % loss.data.item())
    def w_distance(p,q):
        return (torch.mean(critic(p)) - torch.mean(critic(q))).item()
    return w_distance



############## QUESTION 3 ###############
def q3():
    random.seed(0)
    trandom.manual_seed(0)
    d_p = distribution1(0, 512)

    ws_distances = []
    js_divergences = []

    for phi in [x/10. for x in range(-10,11)]:
        d_q = distribution1(phi, 512)
        train = [(next(d_p), next(d_q)) for i in range(500)]
        critic = q2(train)
        ws_distances.append((phi, critic(next(d_p),next(d_q))))
        print("Phi %.3f, WS %.3f" % (ws_distances[-1][0], ws_distances[-1][1]))
    
    for phi in [x/10. for x in range(-10,11)]:
        d_q = distribution1(phi, 512)
        train = [(next(d_p), next(d_q)) for i in range(300)]
        discriminator = q1(train)
        js_divergences.append((phi, discriminator(next(d_p), next(d_q))))
        print("Phi %.3f, JS %.3f" % (js_divergences[-1][0], js_divergences[-1][1]))
    
    print(ws_distances)
    f = open("ws.tab", "w")
    f.write("\n".join(["%.3f\t%.3f" % (x,y) for (x,y) in ws_distances]))
    f.close()
    print(js_divergences)
    f = open("js.tab", "w")
    f.write("\n".join(["%.3f\t%.3f" % (x,y) for (x,y) in js_divergences]))
    f.close()



############## QUESTION 4 ###############
def q4():
    def create_discriminator(data):
        assert (data[0][0].size()[1] == data[0][1].size()[1])
        batch_size = data[0][0].size()[0]
        input_size = data[0][0].size()[1]
        print("Input size %s:%s" % data[0][0].size())
        discriminator = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        discriminator.double()
        optimizer = optim.SGD(discriminator.parameters(), lr=0.001)
        for epoch in range(10):
            #print("Q4 Epoch #%s" % (epoch+1))
            for (p, q) in data:
                discriminator.zero_grad()
                loss = -(torch.mean(torch.log(discriminator(p))) + torch.mean(torch.log(1 - discriminator(q))))
                loss.backward()
                optimizer.step()
        #print("Q4 Loss :%s" % loss.data.item())
        return discriminator

    def train_discriminator():
        d_p = distribution4(512) # f1
        d_q = distribution3(512) # f0
        train = [(next(d_p), next(d_q)) for i in range(100)]
        return create_discriminator(train)

    discriminator = train_discriminator()
    f0 = next(distribution3(512))
    def estimated_f1(x):
        #return (f0 * discriminator(x))/(1 - discriminator(x))
        return (discriminator(x))/(1 - discriminator(x))

    def to_batch(xx):
        return torch.from_numpy(np.array([[x] for x in xx]))

    def plot():
        xx = np.linspace(-5,5,1000)
        #xx = torch.randn(10000).double().numpy()
        f = lambda x: torch.tanh(x*2+1) + x*0.75
        d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
        N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
        #bbb = discriminator(torch.from_numpy(np.array([[x] for x in xx])))
        #print(bbb)
        r = discriminator(to_batch(xx)) # evaluate xx using your discriminator; replace xx with the output
        #print(r)
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(xx,r.detach().numpy())
        plt.title(r'$D(x)$')
        plt.savefig("q1.4.png")
        estimate = (f0 * estimated_f1(to_batch(xx))).detach().numpy() # estimate the density of distribution4 (on xx) using the discriminator; replace "np.ones_like(xx)*0." with your estimate
        plt.subplot(1,2,2)
        plt.plot(xx,estimate)

        plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
        plt.legend(['Estimated','True'])
        plt.title('Estimated vs True')
        plt.savefig("q1.4.png")

    plot()





    #plot discriminator
    #plit estimated_f1

    #p : f1 : distribution4
    #q : f0 : gaussian
    #f1(x) = (f0(x) * d(x))/(1 - d(x))


if __name__ == "__main__":
    q4()