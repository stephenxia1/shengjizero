#!/usr/bin/env python3
import os
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from env1 import Env1
from dqn  import QNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)


def train_dqn(episodes=10000, batch_size=64, gamma=0.99, lr=1e-3,
              buffer_size=10000, target_update=100, eps_start=1.0,
              eps_end=0.05, eps_decay=500):
    env = Env1()
    state_dim  = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)

    eps = eps_start
    returns = []

    for ep in range(1, episodes+1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
            
            if random.random() < eps:
                valid = env.getValidActions(env.current_player)
                action = int(np.random.choice(np.where(valid==1)[0]))
            else:
                with torch.no_grad():
                    q = policy_net(state)
                mask = torch.from_numpy(env.getValidActions(env.current_player).astype(bool)).unsqueeze(0)
                q[~mask] = -float('inf')
                action = int(q.argmax(1).item())

            next_obs, _, done, _ = env.step(action)
            reward = env.get_reward(env.current_player)

            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            if len(replay) >= batch_size:
                s_b, a_b, r_b, ns_b, d_b = replay.sample(batch_size)
                s_b  = torch.from_numpy(s_b.astype(np.float32))
                ns_b = torch.from_numpy(ns_b.astype(np.float32))
                a_b  = torch.from_numpy(a_b).long().unsqueeze(1)
                r_b  = torch.from_numpy(r_b.astype(np.float32))
                d_b  = torch.from_numpy(d_b.astype(np.float32))

                q_vals = policy_net(s_b).gather(1, a_b).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(ns_b).max(1)[0]
                    q_target = r_b + (1 - d_b) * gamma * q_next

                loss = torch.nn.functional.mse_loss(q_vals, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eps = max(eps_end, eps * np.exp(-1.0/eps_decay))
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        returns.append(total_reward)
        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} Return={total_reward:.2f} Eps={eps:.2f}")

    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    print('Training complete, saved dqn_model.pth')

    plt.figure()
    plt.plot(range(1, episodes+1), returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Curve')
    plt.tight_layout()
    plt.savefig('training_curve.png')
    print('Saved training_curve.png')


if __name__ == '__main__':
    train_dqn()
