#!/usr/bin/env python3
import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

from jointqnlstm import JointDQNLSTM  # assuming the joint LSTM class is in jointqnlstm.py
from env1 import Env1

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_joint_lstm(model_path, episodes=100):
    env = Env1()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found.")
    model = JointDQNLSTM(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Containers
    team0_returns = []      # returns when seats 0&2 are landlord
    team1_returns = []      # returns when seats 1&3 are landlord
    detailed      = []      # per-episode team return
    flags         = []      # landlord parity
    team0_first   = []      # first-play cards when seats0&2 landlord
    team1_first   = []      # first-play cards when seats1&3 landlord

    for ep in range(1, episodes+1):
        env.reset()
        obs0 = env.getObs(0)
        obs2 = env.getObs(2)
        flag = (env.landlord % 2)
        flags.append(flag)
        first_card = None
        done = False
        h0 = None; h2 = None

        while not done:
            player = env.current_player
            if player in (0, 2):
                s0 = torch.from_numpy(obs0).float().unsqueeze(0).unsqueeze(1).to(device)
                s2 = torch.from_numpy(obs2).float().unsqueeze(0).unsqueeze(1).to(device)
                with torch.no_grad():
                    qgrid, h0, h2 = model(s0, s2, h0, h2)
                grid = qgrid.cpu().numpy()[0]
                valid0 = env.getValidActions(0).astype(bool)
                valid2 = env.getValidActions(2).astype(bool)
                mask = np.outer(valid0, valid2)
                grid[~mask] = -np.inf
                a0, a2 = np.unravel_index(np.argmax(grid), grid.shape)
                action = a0 if player==0 else a2
            else:
                valid = env.getValidActions(player)
                opts  = np.where(valid==1)[0]
                action = int(random.choice(opts)) if opts.size else 0

            if first_card is None:
                first_card = action

            # ensure selected action is valid
            valid_cur = env.getValidActions(player).astype(bool)
            if not valid_cur[action]:
                valid_idxs = np.where(valid_cur)[0]
                action = int(np.random.choice(valid_idxs)) if valid_idxs.size else 0

            env.step(action)
            done = env.done
            obs0 = env.getObs(0)
            obs2 = env.getObs(2)

        # end episode: collect returns and first card
        if flag==0:
            team_ret = env.get_reward(1) + env.get_reward(3)
            team0_returns.append(team_ret)
            team0_first.append(first_card)
        else:
            team_ret = env.get_reward(0) + env.get_reward(2)
            team1_returns.append(team_ret)
            team1_first.append(first_card)
        detailed.append(team_ret)
        print(f"Episode {ep}/{episodes}: team return={team_ret:.2f}")

    # Stats
    m0, s0 = (np.mean(team0_returns), np.std(team0_returns)) if team0_returns else (0,0)
    m1, s1 = (np.mean(team1_returns), np.std(team1_returns)) if team1_returns else (0,0)
    print(f"Seats1&3 landlord: mean={m0:.2f}, std={s0:.2f}, count={len(team0_returns)}")
    print(f"Seats0&2 landlord: mean={m1:.2f}, std={s1:.2f}, count={len(team1_returns)}")

        # Bar chart of mean returns by landlord team
    labels = ['Seats 1 & 3 (Random)', 'Seats 0 & 2 (LSTM)']
    means = [m0, m1]
    errs = [s0, s1]
    x = np.arange(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Team Return')
    plt.title('Joint LSTM Mean Returns by Landlord Team')
    plt.tight_layout()
    out_bar = 'joint_lstm_mean_returns.png'
    plt.savefig(out_bar)
    print(f"Saved bar chart to {out_bar}")
    plt.show()


    # Sort first-play cards by rank per team
    rank_order = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','Joker']
    def rank_label(idx):
        if idx>=52: return 'Joker'
        if idx>=48: return '2'
        r=idx%12
        return {11:'A',10:'K',9:'Q',8:'J',7:'10'}.get(r,str(r+3))

    # Frequency plot of first-play card ranks per team
    freq0 = Counter(rank_label(c) for c in team0_first)
    freq1 = Counter(rank_label(c) for c in team1_first)
    ranks = rank_order
    counts0 = [freq0[r] for r in ranks]
    counts1 = [freq1[r] for r in ranks]
    x = np.arange(len(ranks))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts0, width, label='Seats0&2')
    plt.bar(x + width/2, counts1, width, label='Seats1&3')
    plt.xticks(x, ranks, rotation=45)
    plt.ylabel('Frequency')
    plt.title('First-play Card Rank Frequency by Team')
    plt.legend()
    plt.tight_layout()
    out3 = 'joint_lstm_first_rank_freq.png'
    plt.savefig(out3)
    print(f"Saved frequency plot to {out3}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='joint_lstm_dqn.pth')
    parser.add_argument('--episodes',   type=int, default=5000)
    args = parser.parse_args()
    evaluate_joint_lstm(args.model_path, args.episodes)
