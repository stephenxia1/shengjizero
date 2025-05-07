

#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from dqn     import QNetwork
from dqnlstm import DQNLSTM
from env1    import Env1


def load_model_auto(path, state_dim, action_dim):
    sd = torch.load(path, map_location='cpu')
    if any(k.startswith('lstm.weight_ih_l0') for k in sd.keys()):
        hidden_dim = sd['lstm.weight_ih_l0'].shape[0] // 4
        model = DQNLSTM(state_dim, hidden_dim, action_dim)
    else:
        model = QNetwork(state_dim, action_dim)
    model.load_state_dict(sd)
    model.eval()
    return model


def evaluate_model(model, env_cls, episodes=5000):
    
    returns = []
    leads   = []
    flags   = []
    is_lstm = isinstance(model, DQNLSTM)

    for _ in range(episodes):
        env = env_cls()
        obs = env.reset()
        flags.append(1 - env.landlord % 2)
        done = False
        hidden = None

        # placeholder for first trick detection
        first_trick_played = False

        while not done:
            new_trick = (env.current_trick.sum() == 0)
            vec = np.array(obs, dtype=np.float32).flatten()
            t   = torch.from_numpy(vec).unsqueeze(0)

            with torch.no_grad():
                if is_lstm:
                    q, hidden = model(t.unsqueeze(1), hidden)
                else:
                    q = model(t)

            valid = env.getValidActions(env.current_player).astype(bool)
            mask  = torch.from_numpy(valid).unsqueeze(0)
            q[~mask] = -float('inf')
            action = int(q.argmax(dim=1).item())

            if new_trick and not first_trick_played:
                leads.append(action)
                first_trick_played = True

            obs, _, done, _ = env.step(action)
            if done:
                # collect all four players' final rewards
                r = tuple(env.get_reward(p) for p in range(4))
                returns.append(r)

    return returns, leads, flags


def main():
    sample_env = Env1()
    state_dim  = int(np.prod(sample_env.observation_space.shape))
    action_dim = int(sample_env.action_space.n)

    # gather QNetwork/LSTM checkpoints
    configs = []
    for f in sorted(glob.glob('*.pth')):
        try:
            sd = torch.load(f, map_location='cpu')
            if any(k.startswith('net.0') for k in sd.keys()) or any(k.startswith('lstm.weight_ih_l0') for k in sd.keys()):
                configs.append((os.path.splitext(f)[0], f))
        except:
            continue
    if not configs:
        print("No QNetwork/LSTM .pth files found.")
        return

    episodes = 100
    stats   = []
    all_leads = []

    for name, path in configs:
        print(f"Evaluating {name}...")
        model = load_model_auto(path, state_dim, action_dim)
        returns, leads, flags = evaluate_model(model, Env1, episodes)
        all_leads.extend(leads)

        
        team0 = []  
        team1 = []  
        for (r0,r1,r2,r3), f in zip(returns, flags):
            if f == 0:
                team0.append(r0 + r2)
            else:
                team1.append(r1 + r3)

        c0, c1 = len(team0), len(team1)
        m0, s0 = (np.mean(team0), np.std(team0)) if c0 else (0,0)
        m1, s1 = (np.mean(team1), np.std(team1)) if c1 else (0,0)
        stats.append((name, c0, m0, s0, c1, m1, s1))

        print(f"  {name}: team0_eps={c0}, mean0={m0:.2f}, std0={s0:.2f}; "
              f"team1_eps={c1}, mean1={m1:.2f}, std1={s1:.2f}")

        
        ep_returns = [ (r0+r2) if f==0 else (r1+r3)
                       for (r0,r1,r2,r3), f in zip(returns, flags) ]
        plt.figure()
        plt.plot(range(1, episodes+1), ep_returns, marker='o')
        plt.title(f"{name} Team Returns (landlord episodes unified)")
        plt.xlabel('Episode')
        plt.ylabel('Team Return')
        plt.tight_layout()
        plt.savefig(f"{name}_team_returns.png")
        plt.close()

        

    seat_returns = {p: [] for p in range(4)}
    for (r0, r1, r2, r3), f in zip(returns, flags):
        if f == 0:
            seat_returns[0].append(r0)
            seat_returns[2].append(r2)
        else:
            seat_returns[1].append(r1)
            seat_returns[3].append(r3)

    means = [np.mean(seat_returns[p]) if seat_returns[p] else 0.0 for p in range(4)]
    stds  = [np.std(seat_returns[p])  if seat_returns[p] else 0.0 for p in range(4)]

    labels = [f"Seat {p}" for p in range(4)]
    x = np.arange(4)
    plt.figure(figsize=(8,5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Return (landlord episodes)')
    plt.title('Per-Seat Returns on Landlord Episodes')
    plt.tight_layout()
    plt.savefig('per_seat_comparison.png')
    plt.show()

    rank_labels = ['A','K','Q','J','10','9','8','7','6','5','4','3','2','Joker']
    rank_map = {}
    for idx in range(54):
        if idx >= 52:    rank_map[idx] = 'Joker'
        elif idx >= 48:  rank_map[idx] = '2'
        else:
            r = idx % 12; mp={11:'A',10:'K',9:'Q',8:'J',7:'10'}
            rank_map[idx] = mp.get(r, str(r+3))
    freq = {lbl:0 for lbl in rank_labels}
    for c in all_leads:
        freq[rank_map[c]] += 1
    plt.figure(figsize=(10,5))
    plt.bar(rank_labels, [freq[l] for l in rank_labels])
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Lead Rank Frequency')
    plt.tight_layout()
    plt.savefig('lead_rank_freq.png')
    plt.show()

if __name__ == '__main__':
    main()

