# Regular DQN vs Regular LSTM
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from dqnlstm import DQNLSTM
from dqn import QNetwork
from env1 import Env1
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_mixed_joint(path1, path2, episodes):
    env = Env1()
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    
    lstm = DQNLSTM(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    lstm.load_state_dict(torch.load(path1, map_location=device))
    lstm.eval()

    
    dqn = QNetwork(state_dim, action_dim).to(device)
    dqn.load_state_dict(torch.load(path2, map_location=device))
    dqn.eval()

    team02_returns = []
    team13_returns = []
    flags         = []     
    first02       = []     
    first13       = []    

   
    init_hand_pts_02 = []  
    earned_pts_02   = []
    init_hand_pts_13 = []  
    earned_pts_13   = []
    first02 = []        
    first13 = []        

    for ep in range(1, episodes+1):
        obs = env.reset()
        h1 = None
        h3 = None
        first_card = None
        landlord_flag = env.landlord % 2
        flags.append(landlord_flag)
        hidden = {0: None, 2: None}
        hidden[0] = None
        hidden[2] = None

        obs0 = env.getObs(0)
        obs2 = env.getObs(2)
        h1 = h3 = None

        first_card = None

        if flags[-1] == 0:
            pts_init = env.getPoints(env.hands[1]) + env.getPoints(env.hands[3])
            init_hand_pts_13.append(pts_init)
        else:
            pts_init = env.getPoints(env.hands[0]) + env.getPoints(env.hands[2])
            init_hand_pts_02.append(pts_init)
        

        done = False
        while not done:
            player = env.current_player
            valid = env.getValidActions(player).astype(bool)
            if player in (0, 2):
                obs_p = env.getObs(player)
                t = torch.from_numpy(obs_p.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(device)
                with torch.no_grad():
                    qvals, hidden[player] = lstm(t, hidden[player])
                q = qvals.cpu().numpy().flatten()
                q[~valid] = -np.inf
                action = int(np.argmax(q))
            else:
                vec = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = dqn(vec).cpu().numpy().flatten()

                valid = env.getValidActions(player).astype(bool)
                q[~valid] = -np.inf
                action = int(np.argmax(q))

            if first_card is None and player == env.landlord:
                first_card = action
            valid = env.getValidActions(player).astype(bool)
            if not valid[action]:
                opts   = np.where(valid)[0]
                action = int(random.choice(opts)) if opts.size else 0

            env.step(action)
            done = env.done
            obs0 = env.getObs(0)
            obs2 = env.getObs(2)

        rewards = [env.get_reward(i) for i in range(4)]
        t02 = rewards[0]
        t13 = rewards[1]
        team02_returns.append(t02)
        team13_returns.append(t13)

        if first_card is not None:
            if landlord_flag == 0:
                first02.append(first_card)
            else:
                first13.append(first_card)

        print(f"Episode {ep}/{episodes}: T02={t02:.2f}, T13={t13:.2f}, first={first_card}")

        if first_card is not None:
            (first02 if flags[-1]==0 else first13).append(first_card)

        raw_pts = env.points
        if flags[-1] == 0:
            earned_pts_13.append(raw_pts)
        else:
            earned_pts_02.append(raw_pts)



    lr02 = [r for r,f in zip(team02_returns, flags) if f==1]
    lr13 = [r for r,f in zip(team13_returns, flags) if f==0]
    m02, s02 = (np.mean(lr02), np.std(lr02)) if lr02 else (0,0)
    m13, s13 = (np.mean(lr13), np.std(lr13)) if lr13 else (0,0)

    print(f"\nTeam0&2 eps: mean={m02:.2f}, std={s02:.2f}, count={len(lr02)}")
    print(f"Team1&3 eps: mean={m13:.2f}, std={s13:.2f}, count={len(lr13)}")


    labels = ['Team 0&2 (LSTM)', 'Team 1&3 (DQN)']
    means  = [m02, m13]
    errs   = [s02, s13]
    x = np.arange(2)
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Return')
    plt.title('LSTM vs DQN')
    plt.tight_layout()
    for xi, (mu, sd) in enumerate(zip(means, errs)):
        plt.text(xi, mu + sd + 0.02, f"{mu:.2f}±{sd:.2f}",
                 ha='center', va='bottom', fontsize=10)
    plt.savefig('lstmdqn_bar_filtered.png')
    plt.show()

    initavg02 = np.mean(init_hand_pts_02) if init_hand_pts_02 else 0
    endavg02 = np.mean(earned_pts_02) if earned_pts_02 else 0
    initavg13 = np.mean(init_hand_pts_13) if init_hand_pts_13 else 0
    endavg13 = np.mean(earned_pts_13) if earned_pts_13 else 0
    x_pts = np.arange(2)
    width=0.4
    plt.figure(figsize=(8,5))
    bars_i = plt.bar(x_pts-width/2,[initavg02,initavg13],width,label='Init Pts')
    bars_e = plt.bar(x_pts+width/2,[endavg02,endavg13],width,label='Earned Pts')
    plt.xticks(x_pts,['LSTM','DQN'])
    plt.ylabel('Points')
    plt.title('Points in Hand vs Points Earned: LSTM vs DQN')
    for b,var in zip(bars_i, ['Init Pts','Init Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    for b,var in zip(bars_e, ['End Pts','End Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    plt.legend(); plt.tight_layout(); plt.savefig('lstmdqn_points_compare.png'); plt.show()

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
    plt.bar(x_rank - width/2, counts02, width, label='First-card LSTM DQN')
    plt.bar(x_rank + width/2, counts13, width, label='First-card DQN')
    plt.xticks(x_rank, ranks, rotation=45)
    plt.xlabel('Card Rank')
    plt.ylabel('Frequency')
    plt.title('First-play Card Rank Frequency by Landlord Team: LSTM vs DQN')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lstmdqn_first_rank_freq.png')
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm', default='dqnlstm_model.pth')
    parser.add_argument('--dqn', default='dqn_model.pth')
    parser.add_argument('--episodes',  type=int, default=5000)
    args = parser.parse_args()
    evaluate_mixed_joint(args.lstm, args.dqn, args.episodes)
