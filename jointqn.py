
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from env1 import Env1

class JointQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_layers=1):
        super(JointQNetwork, self).__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(2 * state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim * action_dim)

    def forward(self, s0, s2):
        x = torch.cat([s0, s2], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.out(x) 
        return q.view(-1, self.action_dim, self.action_dim)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s0, s2, a0, a2, r, ns0, ns2, done):
        self.buffer.append((s0, s2, a0, a2, r, ns0, ns2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s0, s2, a0, a2, r, ns0, ns2, d = map(np.array, zip(*batch))
        return s0, s2, a0, a2, r, ns0, ns2, d

    def __len__(self):
        return len(self.buffer)


def train_joint_dqn(
    episodes,
    model_path,
    hidden_dim=256,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    target_update=500,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=10000
):
    env = Env1()
    state_dim  = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    policy = JointQNetwork(state_dim, hidden_dim, action_dim)
    target = JointQNetwork(state_dim, hidden_dim, action_dim)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)

    steps = 0
    eps = eps_start

    for ep in range(1, episodes+1):
        env.reset()
        obs0 = env.getObs(0)
        obs2 = env.getObs(2)
        done = False
        team_return = 0.0

        while not done:
            player = env.current_player
            if player in (0,2):
                if random.random() < eps:
                    va0 = np.where(env.getValidActions(0)==1)[0]
                    va2 = np.where(env.getValidActions(2)==1)[0]
                    a0 = int(random.choice(va0)) if va0.size else random.randrange(action_dim)
                    a2 = int(random.choice(va2)) if va2.size else random.randrange(action_dim)
                else:
                    s0_t = torch.from_numpy(obs0).unsqueeze(0).float()
                    s2_t = torch.from_numpy(obs2).unsqueeze(0).float()
                    with torch.no_grad():
                        qjoint = policy(s0_t, s2_t).cpu().numpy()[0]
                    mask = np.outer(env.getValidActions(0).astype(bool),
                                    env.getValidActions(2).astype(bool))
                    qjoint[~mask] = -np.inf
                    a0, a2 = np.unravel_index(qjoint.argmax(), qjoint.shape)
                action = a0 if player==0 else a2
                valid_actions = np.where(env.getValidActions(player)==1)[0]
                if action not in valid_actions:
                    if valid_actions.size > 0:
                        action = int(random.choice(valid_actions))
                    else:
                        action = 0 

                env.step(action)
                done = env.done
                if done:
                    team_return = env.get_reward(0) + env.get_reward(2)
                ns0, ns2 = env.getObs(0), env.getObs(2)
                replay.push(obs0, obs2, a0, a2,
                            team_return if done else 0,
                            ns0, ns2,
                            done)
                obs0, obs2 = ns0, ns2

                if len(replay) >= batch_size:
                    s0_b, s2_b, a0_b, a2_b, r_b, ns0_b, ns2_b, d_b = replay.sample(batch_size)
                    s0_b = torch.from_numpy(s0_b).float()
                    s2_b = torch.from_numpy(s2_b).float()
                    ns0_b = torch.from_numpy(ns0_b).float()
                    ns2_b = torch.from_numpy(ns2_b).float()
                    a0_b = torch.from_numpy(a0_b).long()
                    a2_b = torch.from_numpy(a2_b).long()
                    r_b  = torch.from_numpy(r_b).float()
                    d_b  = torch.from_numpy(d_b.astype(np.float32)).float()

                    q_pred = policy(s0_b, s2_b)
                    q_pred = q_pred[torch.arange(batch_size), a0_b, a2_b]

                    with torch.no_grad():
                        q_next = target(ns0_b, ns2_b)
                        q_next = q_next.view(batch_size, -1).max(1)[0]
                        q_target = r_b + (1 - d_b) * gamma * q_next

                    loss = nn.MSELoss()(q_pred, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                steps += 1
                eps = max(eps_end, eps_start * np.exp(-steps / eps_decay))
                if steps % target_update == 0:
                    target.load_state_dict(policy.state_dict())

            else:
                valid = env.getValidActions(player)
                choices = np.where(valid == 1)[0]
                if choices.size == 0:
                    choices = np.arange(action_dim)
                action = int(random.choice(choices))
                env.step(action)
                done = env.done

        if ep % 50 == 0:
            print(f"Episode {ep}/{episodes}, Team Return={team_return:.2f}, eps={eps:.3f}")

    torch.save(policy.state_dict(), model_path)
    print(f"Training complete. Joint model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',   type=int, default=20000)
    parser.add_argument('--model-path', type=str, default='joint_team_dqn.pth')
    args = parser.parse_args()
    train_joint_dqn(args.episodes, args.model_path)
