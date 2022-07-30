import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
from env import Game

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sars = namedtuple('sars', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add_memory(self, state, action, reward, next_state):
        self.memory.append(sars(state, action, reward, next_state))

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNModel(nn.Module):
    def __init__(self, w, h, lr, num_actions):
        super(DQNModel, self).__init__()
        self.to(DEVICE)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_actions)

    def forward(self, state):
        x = x.to(DEVICE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Agent:
    def __init__(self, w, h, lr, eps, discount, batch_size, num_actions, max_memory = 1e6,
                eps_min = 0.01, eps_dec = 1e-4):
        self.lr = lr
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.discount = discount
        self.batch_size = batch_size
        self.action_space = np.arange(num_actions)
        self.max_memory = max_memory

        self.QEvaluation = DQNModel(w=w, h=h, lr=lr, num_actions=num_actions)
        self.replayMemory = ReplayMemory(max_memory)

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = torch.tensor([observation]).to(DEVICE) # need wrapper?
            actions = self.QEvaluation.forward(state)
            chosen_action = torch.argmax(actions).item()
        else:
            chosen_action = np.random.choice(self.action_space)
        return chosen_action
    
    def store_transition(self, state, action, reward, next_state):
        self.replayMemory.add_memory(state, action, reward, next_state)

    def learn(self):
        if len(self.replayMemory) < self.max_memory:
            return
        
        sars = self.replayMemory.random_sample(self.batch_size)
        batch = ReplayMemory(*zip(*sars)) # convert batch-array of sars to sars of batch-arrays

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        q_eval = self.QEvaluation.forward(state_batch).gather(1, action_batch)

        q_next = self.QEvaluation.forward(next_state_batch)
        q_target = reward_batch + self.discount * q_next

        self.QEvaluation.optimizer.zero_grad()
        loss = self.QEvaluation.loss(q_target, q_eval).to(DEVICE)
        loss.backward()
        self.QEvaluation.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

if __name__ == '__main__':
    env = Game(5)
    agent = Agent(w=env.grid_size, h=env.grid_size, lr=0.001, eps=0.99, discount=0.99, batch_size=64, num_actions=4)
    scores, eps_history = [], []
    num_games = 1e6
    
    for i in range(num_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.eps)
        
        avg_score = np.mean(scores[-100:])

        