import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
from env import Game

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Sars = namedtuple('Sars', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add_memory(self, state, action, reward, next_state):
        self.memory.append(Sars(state, action, reward, next_state))

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNModel(nn.Module):
    def __init__(self, w, h, lr, num_actions):
        super(DQNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)

        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))

        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, num_actions) # num_actions is 4 here?

        self.loss = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, w, h, lr, eps, discount, batch_size, num_actions, max_memory = 100000,
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
        direction = ('right', 'left', 'up', 'down')
        if np.random.random() > self.eps:
            state = observation.to(DEVICE)
            actions = self.QEvaluation.forward(state)
            chosen_action = torch.argmax(actions).item()
        else:
            chosen_action = np.random.choice(self.action_space)
        return direction[chosen_action]
    
    def store_transition(self, state, action, reward, next_state):
        one_hot_action = torch.zeros(4)
        mapping = {'right': 0, 'left': 1, 'up': 2, 'down': 3}
        one_hot_action[mapping[action]] = 1
        self.replayMemory.add_memory(state, one_hot_action, reward, next_state)

    def learn(self):
        if len(self.replayMemory) < self.max_memory:
            return
        
        transitions = self.replayMemory.random_sample(self.batch_size)
        batch = Sars(*zip(*transitions)) # convert batch-array of Sars to Sars of batch-arrays

        non_final_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.next_state)), dtype=bool, device=DEVICE)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_eval = self.QEvaluation.forward(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        next_state_values[non_final_mask] = torch.max(self.QEvaluation(non_final_next_states), 1).detach()
        q_target = reward_batch + self.discount * next_state_values

        self.QEvaluation.optimizer.zero_grad()
        loss = self.QEvaluation.loss(q_target, q_eval).to(DEVICE)
        loss.backward()
        self.QEvaluation.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

if __name__ == '__main__':
    env = Game(5)
    agent = Agent(w=env.grid_size, h=env.grid_size, lr=0.001, eps=0.99, discount=0.99, batch_size=64, num_actions=4)
    scores, eps_history = [], []
    num_games = 1000000

    for i in range(num_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            agent.store_transition(observation, action, torch.tensor([reward], device=DEVICE), observation_)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.eps)
        
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score, ' average score %.1f' % avg_score, 'epsilon %.2f' % agent.eps)
        