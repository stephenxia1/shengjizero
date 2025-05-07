#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from jointqn import JointQNetwork
from env1    import Env1


def evaluate_joint_dqn(model_path, episodes=1000):
    """
    Evaluate a trained JointQNetwork over a specified number of episodes.
    Agents 0&2 use the joint-Q policy; agents 1&3 act randomly.

    Tracks per-episode team returns and uses flags to compute
    mean/std only on episodes where team 0&2 was landlord.
    """
    env = Env1()
    state_dim  = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    # Load network
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
    policy = JointQNetwork(state_dim, hidden_dim=256, action_dim=action_dim)
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()

    team0_returns = []  # sum r0+r2 when seats 0&2 landlord
    team1_returns = []  # sum r1+r3 when seats 1&3 landlord
    flags = []         # 0 if seats0&2 landlord, 1 otherwise
    detailed = []      # list of per-episode team return

    for ep in range(1, episodes+1):
        env.reset()
        obs0 = env.getObs(0)
        obs2 = env.getObs(2)
        flags.append(env.landlord % 2)
        done = False

        # play episode
        while not done:
            player = env.current_player
            if player in (0,2):
                s0_t = torch.from_numpy(obs0).unsqueeze(0).float()
                s2_t = torch.from_numpy(obs2).unsqueeze(0).float()
                with torch.no_grad():
                    qjoint = policy(s0_t, s2_t).cpu().numpy()[0]
                v0 = env.getValidActions(0).astype(bool)
                v2 = env.getValidActions(2).astype(bool)
                mask = np.outer(v0, v2)
                qjoint[~mask] = -np.inf
                a0,a2 = np.unravel_index(np.argmax(qjoint), qjoint.shape)
                action = a0 if player==0 else a2
            else:
                valid = env.getValidActions(player)
                opts  = np.where(valid==1)[0]
                action = int(random.choice(opts)) if opts.size else 0

            # ensure validity
            cur_valid = env.getValidActions(player)
            if not cur_valid[action]:
                choices = np.where(cur_valid==1)[0]
                action = int(random.choice(choices)) if choices.size else 0

            env.step(action)
            done = env.done
            obs0 = env.getObs(0)
            obs2 = env.getObs(2)

        # end of episode: collect team returns
        r0 = env.get_reward(0)
        r1 = env.get_reward(1)
        r2 = env.get_reward(2)
        r3 = env.get_reward(3)
        if flags[-1] == 0:
            team0_returns.append(r0 + r2)
            detailed.append(r0 + r2)
        else:
            team1_returns.append(r1 + r3)
            detailed.append(r1 + r3)

        print(f"Episode {ep}/{episodes}: team return = {detailed[-1]:.2f}")

    # Compute filtered statistics
    mean0 = np.mean(team0_returns) if team0_returns else 0.0
    std0  = np.std(team0_returns)  if team0_returns else 0.0
    mean1 = np.mean(team1_returns) if team1_returns else 0.0
    std1  = np.std(team1_returns)  if team1_returns else 0.0

    print(f"\nEvaluation over {episodes} episodes:")
    print(f"  Seats 0&2 (landlord eps): mean={mean0:.2f}, std={std0:.2f}, count={len(team0_returns)}")
    print(f"  Seats 1&3 (landlord eps): mean={mean1:.2f}, std={std1:.2f}, count={len(team1_returns)}")

    # plot per-episode team returns
    plt.figure(figsize=(8,5))
    plt.plot(range(1, episodes+1), detailed, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Team Return')
    plt.title('JointDQN Team Returns (filtered by landlord)')
    plt.tight_layout()
    plt.savefig('joint_team_returns_filtered.png')
    print("Saved filtered team return plot.")
    plt.close()

    # bar chart per team
    labels = ['Seats 0&2','Seats 1&3']
    means = [mean0, mean1]
    errs  = [std0, std1]
    x = np.arange(2)
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Team Return')
    plt.title('JointDQN Team Performance on Landlord Episodes')
    plt.tight_layout()
    plt.savefig('joint_team_bar_filtered.png')
    print("Saved filtered team bar chart.")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='joint_team_dqn.pth')
    parser.add_argument('--episodes',   type=int, default=1000)
    args = parser.parse_args()
    evaluate_joint_dqn(args.model_path, args.episodes)