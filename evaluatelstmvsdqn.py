#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from jointqn import JointQNetwork
from jointqnlstm import JointDQNLSTM
from env1 import Env1
from collections import Counter

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_mixed_joint(q_path, lstm_path, episodes=500):
    env = Env1()
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    # Load Joint-Q network (for players 0 & 2)
    joint_q = JointQNetwork(state_dim, hidden_dim=256, action_dim=action_dim).to(device)
    joint_q.load_state_dict(torch.load(q_path, map_location=device))
    joint_q.eval()

    # Load Joint-LSTM-Q network (for players 1 & 3)
    joint_lstm = JointDQNLSTM(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    joint_lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    joint_lstm.eval()

    team02_returns = []
    team13_returns = []
    flags         = []     # landlord parity: 0 if team0&2 was landlord
    first02       = []     # first-play cards when team0&2 landlord
    first13       = []     # first-play cards when team1&3 landlord

    for ep in range(1, episodes+1):
        env.reset()
        h1 = None
        h3 = None
        first_card = None
        landlord_flag = env.landlord % 2
        flags.append(landlord_flag)

        obs0 = env.getObs(0)
        obs2 = env.getObs(2)
        h1 = h3 = None

        first_card = None
        done = False
        while not done:
            player = env.current_player
            # always advance LSTM for seats 1 & 3, using zero‐input for the one that didn't play
            t_player = torch.from_numpy(env.getObs(player)).float() \
                               .unsqueeze(0).unsqueeze(1).to(device)
            inp1 = t_player if player == 1 else torch.zeros_like(t_player)
            inp3 = t_player if player == 3 else torch.zeros_like(t_player)
            with torch.no_grad():
                _, h1, h3 = joint_lstm(inp1, inp3, h1, h3)
            # pick action
            if player in (0,2):
                s0 = torch.from_numpy(obs0).float().unsqueeze(0).to(device)
                s2 = torch.from_numpy(obs2).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    qgrid = joint_q(s0, s2).cpu().numpy()[0]
                mask = np.outer(env.getValidActions(0).astype(bool),
                                env.getValidActions(2).astype(bool))
                qgrid[~mask] = -np.inf
                a0, a2 = np.unravel_index(qgrid.argmax(), qgrid.shape)
                action = a0 if player==0 else a2
            else:
                s1 = torch.from_numpy(env.getObs(1)).float().unsqueeze(0).unsqueeze(1).to(device)
                s3 = torch.from_numpy(env.getObs(3)).float().unsqueeze(0).unsqueeze(1).to(device)
                inp1 = s1 if player==1 else torch.zeros_like(s1)
                inp3 = s3 if player==3 else torch.zeros_like(s3)
                with torch.no_grad():
                    qgrid, h1, h3 = joint_lstm(inp1, inp3, h1, h3)
                grid = qgrid.cpu().numpy()[0]
                # mask and pick exactly like the standalone
                valid1 = env.getValidActions(1).astype(bool)
                valid3 = env.getValidActions(3).astype(bool)
                mask13 = np.outer(valid1, valid3)
                grid[~mask13] = -np.inf
                a1, a3 = np.unravel_index(grid.argmax(), grid.shape)
                action = a1 if player==1 else a3

            # only record the very first card played by the landlord
            if first_card is None and player == env.landlord:
                first_card = action
            valid = env.getValidActions(player).astype(bool)
            #then validity fallback and stepping:
            if not valid[action]:
                opts   = np.where(valid)[0]
                action = int(random.choice(opts)) if opts.size else 0

            env.step(action)
            done = env.done
            obs0 = env.getObs(0)
            obs2 = env.getObs(2)

        # at episode end, collect returns
        rewards = [env.get_reward(i) for i in range(4)]
        # team02 is players 0&2, team13 is players 1&3
        t02 = rewards[1] + rewards[3]
        t13 = rewards[0] + rewards[2]
        team02_returns.append(t02)
        team13_returns.append(t13)

        # assign first_card to the correct list based on landlord
        if first_card is not None:
            if landlord_flag == 0:
                first02.append(first_card)
            else:
                first13.append(first_card)

        print(f"Episode {ep}/{episodes}: T02={t02:.2f}, T13={t13:.2f}, first={first_card}")

    # filter returns by landlord episodes
    lr02 = [r for r,f in zip(team02_returns, flags) if f==0]
    lr13 = [r for r,f in zip(team13_returns, flags) if f==1]
    m02, s02 = (np.mean(lr02), np.std(lr02)) if lr02 else (0,0)
    m13, s13 = (np.mean(lr13), np.std(lr13)) if lr13 else (0,0)

    print(f"\nTeam0&2 landlord eps: mean={m02:.2f}, std={s02:.2f}, count={len(lr02)}")
    print(f"Team1&3 landlord eps: mean={m13:.2f}, std={s13:.2f}, count={len(lr13)}")

    # bar chart of filtered means
    labels = ['Team 0&2 (DQN)', 'Team 1&3 (LSTM DQN)']
    means  = [m02, m13]
    errs   = [s02, s13]
    x = np.arange(2)
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Return')
    plt.title('Mixed JointQ vs JointLSTM (Landlord Episodes)')
    plt.tight_layout()
    plt.savefig('mixed_joint_bar_filtered.png')
    plt.show()

    # first-play rank frequency
    order = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','Joker']
    def to_label(i):
        if i>=52: return 'Joker'
        if i>=48: return '2'
        return {11:'A',10:'K',9:'Q',8:'J',7:'10'}.get(i%12, str(i%12+3))

    freq02 = Counter(to_label(c) for c in first02)
    freq13 = Counter(to_label(c) for c in first13)
    counts02 = [freq02.get(r,0) for r in order]
    counts13 = [freq13.get(r,0) for r in order]

    plt.figure(figsize=(12,6))
    ranks = order
    N = len(ranks)
    x_rank = np.arange(N)
    width = 0.4

    plt.figure(figsize=(12,6))
    plt.bar(x_rank - width/2, counts02, width, label='First-card when 0&2 landlord (DQN)')
    plt.bar(x_rank + width/2, counts13, width, label='First-card when 1&3 landlord (LSTM DQN)')
    plt.xticks(x_rank, ranks, rotation=45)
    plt.xlabel('Card Rank')
    plt.ylabel('Frequency')
    plt.title('First-play Card Rank Frequency by Landlord Team')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mixed_joint_first_rank_freq.png')
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jointq', default='joint_team_dqn.pth')
    parser.add_argument('--jointlstm', default='joint_lstm_dqn.pth')
    parser.add_argument('--episodes',  type=int, default=5000)
    args = parser.parse_args()
    evaluate_mixed_joint(args.jointq, args.jointlstm, args.episodes)
