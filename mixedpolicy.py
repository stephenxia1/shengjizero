#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from dqn   import QNetwork
from env1  import Env1


def main():
    # Load evaluation env to infer dimensions
    sample_env = Env1()
    state_dim  = int(np.prod(sample_env.observation_space.shape))
    action_dim = int(sample_env.action_space.n)

    # Load trained DQN policy
    dqn_path = 'dqn_model.pth'
    if not os.path.exists(dqn_path):
        print(f"Error: DQN model checkpoint '{dqn_path}' not found.")
        return
    dqn_model = QNetwork(state_dim, action_dim)
    dqn_model.load_state_dict(torch.load(dqn_path, map_location='cpu'))
    dqn_model.eval()

    episodes = 10000
    # track which team was landlord each episode
    flags   = []  # 0 if seats 0&2, 1 if seats 1&3
    returns = {p: [] for p in range(4)}

    for ep in range(episodes):
        env = Env1()
        obs = env.reset()
        # record landlord parity
        flags.append(env.landlord % 2)
        done = False

        while not done:
            player = env.current_player
            valid  = env.getValidActions(player).astype(bool)
            if player in (0, 2):
                # DQN action
                vec = np.array(obs, dtype=np.float32).flatten()
                t   = torch.from_numpy(vec).unsqueeze(0)
                with torch.no_grad(): q = dqn_model(t)
                q = q.cpu().numpy().flatten()
                q[~valid] = -np.inf
                action = int(np.argmax(q))
            else:
                # random action
                choices = np.nonzero(valid)[0]
                action  = int(random.choice(choices)) if choices.size else 0

            obs, _, done, _ = env.step(action)

        # record final rewards
        for p in range(4):
            returns[p].append(env.get_reward(p))

    # filter and compute per-agent stats only on episodes where their team was landlord
    filtered = {p: [] for p in range(4)}
    for i, flag in enumerate(flags):
        for p in range(4):
            team = 0 if p in (0,2) else 1
            if flag == team:
                filtered[p].append(returns[p][i])

    means = {p: np.mean(filtered[p]) if filtered[p] else 0.0 for p in filtered}
    stds  = {p: np.std(filtered[p])  if filtered[p] else 0.0 for p in filtered}

    print(f"Mixed Policy Evaluation over {episodes} episodes (landlord episodes only):")
    for p in range(4):
        count = len(filtered[p])
        print(f" Agent {p}: Count={count}, Mean={means[p]:.2f}, Std={stds[p]:.2f}")

    # per-team aggregated stats
    team0 = filtered[0] + filtered[2]
    team1 = filtered[1] + filtered[3]
    tm_means = [np.mean(team0) if team0 else 0.0,
                np.mean(team1) if team1 else 0.0]
    tm_stds  = [np.std(team0)  if team0 else 0.0,
                np.std(team1)  if team1 else 0.0]

    print("\nTeam-level Returns (landlord episodes):")
    print(f" Team 0&2: Mean={tm_means[0]:.2f}, Std={tm_stds[0]:.2f}")
    print(f" Team 1&3: Mean={tm_means[1]:.2f}, Std={tm_stds[1]:.2f}")

    # plot per-agent
    labels = [f"Agent {p}" for p in range(4)]
    x = np.arange(4)
    plt.figure(figsize=(8,5))
    plt.bar(x, [means[p] for p in range(4)], yerr=[stds[p] for p in range(4)], capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Return')
    plt.title('Per-Agent Returns (landlord episodes)')
    plt.tight_layout()
    plt.savefig('mixed_policy_agents_filtered.png')
    plt.close()

    # plot per-team
    team_labels = ['Seats 0&2','Seats 1&3']
    x2 = np.arange(2)
    plt.figure(figsize=(6,5))
    plt.bar(x2, tm_means, yerr=tm_stds, capsize=5)
    plt.xticks(x2, team_labels)
    plt.ylabel('Mean Return')
    plt.title('Team Returns (landlord episodes)')
    plt.tight_layout()
    plt.savefig('mixed_policy_teams_filtered.png')
    plt.show()

if __name__ == '__main__':
    main()
