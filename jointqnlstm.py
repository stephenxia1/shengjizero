
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env1 import Env1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointDQNLSTM(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_layers=1):
        super(JointDQNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.lstm0 = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True)

        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim * action_dim)

    def forward(self, seq0, seq2, h0=None, h2=None):
        out0, h0 = self.lstm0(seq0, h0)
        out2, h2 = self.lstm2(seq2, h2)
        o0 = out0[:, -1, :]
        o2 = out2[:, -1, :]
        x = torch.relu(self.fc1(torch.cat([o0, o2], dim=-1)))
        x = torch.relu(self.fc2(x))
        q = self.out(x)
        return q.view(-1, self.action_dim, self.action_dim), h0, h2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, seq0, seq2, a0, a2, r, nseq0, nseq2, done):
        self.buffer.append((seq0, seq2, a0, a2, r, nseq0, nseq2, done))

    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        seq0_b, seq2_b, a0_b, a2_b, r_b, ns0_b, ns2_b, d_b = zip(*batch)
        return seq0_b, seq2_b, a0_b, a2_b, r_b, ns0_b, ns2_b, d_b

    def __len__(self):
        return len(self.buffer)


def train_joint_lstm(
    episodes, model_path,
    hidden_dim=128, buffer_size=50000, batch_size=32,
    gamma=0.99, lr=1e-3, target_update=500,
    eps_start=1.0, eps_end=0.05, eps_decay=10000
):
    env = Env1()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = JointDQNLSTM(state_dim, hidden_dim, action_dim).to(device)
    target = JointDQNLSTM(state_dim, hidden_dim, action_dim).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)
    steps = 0
    eps = eps_start

    for ep in range(1, episodes+1):
        env.reset()
        seq0 = [env.getObs(0)]
        seq2 = [env.getObs(2)]
        done = False
        team_return = 0.0
        hidden0 = None
        hidden2 = None

        while not done:
            player = env.current_player
            if player in (0, 2):
                if random.random() < eps:
                    va0 = np.where(env.getValidActions(0) == 1)[0]
                    va2 = np.where(env.getValidActions(2) == 1)[0]
                    a0 = int(random.choice(va0)) if va0.size else random.randrange(action_dim)
                    a2 = int(random.choice(va2)) if va2.size else random.randrange(action_dim)
                else:
                    s0_t = torch.FloatTensor(np.array(seq0)).unsqueeze(0).to(device)
                    s2_t = torch.FloatTensor(np.array(seq2)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        qgrid, hidden0, hidden2 = policy(s0_t, s2_t, hidden0, hidden2)
                    qgrid = qgrid.cpu().numpy()[0]
                    mask = np.outer(env.getValidActions(0).astype(bool), env.getValidActions(2).astype(bool))
                    qgrid[~mask] = -np.inf
                    a0, a2 = np.unravel_index(qgrid.argmax(), qgrid.shape)
                action = a0 if player == 0 else a2

                       
                valid_cur = np.where(env.getValidActions(player)==1)[0]
                if action not in valid_cur:
                    if valid_cur.size > 0:
                        action = int(random.choice(valid_cur))
                    else:
                        action = 0
                env.step(action)
                done = env.done
                if done:
                    team_return = env.get_reward(0) + env.get_reward(2)

       
                ns0 = env.getObs(0)
                ns2 = env.getObs(2)

                
                replay.push(
                    seq0.copy(), seq2.copy(), a0, a2,
                    team_return if done else 0,
                    seq0.copy(), seq2.copy(), done
                )

            
            else:
                valid = env.getValidActions(player)
                opts = np.where(valid == 1)[0]
                action = int(random.choice(opts)) if opts.size else 0
                env.step(action)
                done = env.done

            
            seq0.append(env.getObs(0))
            seq2.append(env.getObs(2))

            
            if len(replay) >= batch_size:
                s0_b, s2_b, a0_b, a2_b, r_b, ns0_b, ns2_b, d_b = replay.sample(batch_size)
                pass

            
            steps += 1
            eps = max(eps_end, eps_start * np.exp(-steps / eps_decay))
            if steps % target_update == 0:
                target.load_state_dict(policy.state_dict())

        if ep % 50 == 0:
            print(f"Ep {ep}/{episodes}, Return={team_return:.2f}, eps={eps:.3f}")

    torch.save(policy.state_dict(), model_path)
    print(f"Saved Joint-LSTM model to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Joint LSTM-DQN')
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--model-path', type=str, default='joint_lstm_dqn.pth')
    args = parser.parse_args()
    train_joint_lstm(args.episodes, args.model_path)
