# LSTM DQN vs Random
import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

from dqnlstm import DQNLSTM 
from env1 import Env1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_lstm(model_path, episodes):
    env = Env1()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found.")
    model = DQNLSTM(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    team0_returns = []      
    team1_returns = []     
    detailed      = []      
    flags         = []   
    team0_first   = []     
    team1_first   = []      

    init_hand_pts_02 = []  
    earned_pts_02   = []
    init_hand_pts_13 = []  
    earned_pts_13   = []
    first02 = []      
    first13 = []       

    hidden = {1: None, 3: None}

    for ep in range(1, episodes+1):
        env.reset()
        flag = (env.landlord % 2)
        flags.append(flag)
        first_card = None
        done = False
        h0 = None; h2 = None
        hidden[0] = None
        hidden[2] = None
        if flag == 0:
            pts_init = env.getPoints(env.hands[1]) + env.getPoints(env.hands[3])
            init_hand_pts_13.append(pts_init)
        else:
            pts_init = env.getPoints(env.hands[0]) + env.getPoints(env.hands[2])
            init_hand_pts_02.append(pts_init)

        while not done:
            player = env.current_player
            valid = env.getValidActions(player).astype(bool)
            if player in (0, 2):
                obs_p = env.getObs(player)
                t = torch.from_numpy(obs_p.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(device)
                with torch.no_grad():
                    qvals, hidden[player] = model(t, hidden[player])
                q = qvals.cpu().numpy().flatten()
                q[~valid] = -np.inf
                action = int(np.argmax(q))
            else:
                valid = env.getValidActions(player)
                opts  = np.where(valid==1)[0]
                action = int(random.choice(opts)) if opts.size else 0

            if first_card is None:
                first_card = action

            valid_cur = env.getValidActions(player).astype(bool)
            if not valid_cur[action]:
                valid_idxs = np.where(valid_cur)[0]
                action = int(np.random.choice(valid_idxs)) if valid_idxs.size else 0

            env.step(action)
            done = env.done

        if flag==0:
            team_ret = env.get_reward(1)
            team1_returns.append(team_ret)
            team0_first.append(first_card)
        else:
            team_ret = env.get_reward(0)
            team0_returns.append(team_ret)
            team1_first.append(first_card)
        detailed.append(team_ret)
        print(f"Episode {ep}/{episodes}: team return={team_ret:.2f}")

        if first_card is not None:
            (first02 if flag==0 else first13).append(first_card)

        raw_pts = env.points
        if flag == 0:
            earned_pts_13.append(raw_pts)
        else:
            earned_pts_02.append(raw_pts)

    m0, s0 = (np.mean(team0_returns), np.std(team0_returns)) if team0_returns else (0,0)
    m1, s1 = (np.mean(team1_returns), np.std(team1_returns)) if team1_returns else (0,0)
    print(f"Seats0&2 landlord: mean={m0:.2f}, std={s0:.2f}, count={len(team0_returns)}")
    print(f"Seats1&3 landlord: mean={m1:.2f}, std={s1:.2f}, count={len(team1_returns)}")

    labels = ['Seats 0 & 2 (LSTM)', 'Seats 1 & 3 (Random)']
    means = [m0, m1]
    errs = [s0, s1]
    
    x = np.arange(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Team Return')
    plt.title('LSTM Mean Returns (Playing Against Landlord)')
    plt.tight_layout()
    for xi, (mu, sd) in enumerate(zip(means, errs)):
        plt.text(xi, mu + sd + 0.02, f"{mu:.2f}±{sd:.2f}",
                 ha='center', va='bottom', fontsize=10)
    out_bar = 'lstmn_mean_returns.png'
    plt.savefig(out_bar)
    print(f"Saved bar chart to {out_bar}")
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
    plt.xticks(x_pts,['LSTM','Random'])
    plt.ylabel('Points')
    plt.title('Points in Hand vs Points Earned: LSTM vs Random')
    for b,var in zip(bars_i, ['Init Pts','Init Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    for b,var in zip(bars_e, ['End Pts','End Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    plt.legend(); plt.tight_layout(); plt.savefig('lstm_points_compare.png'); plt.show()

    rank_order = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','Joker']
    def rank_label(idx):
        if idx>=52: return 'Joker'
        if idx>=48: return '2'
        r=idx%12
        return {11:'A',10:'K',9:'Q',8:'J',7:'10'}.get(r,str(r+3))
    freq0 = Counter(rank_label(c) for c in team0_first)
    freq1 = Counter(rank_label(c) for c in team1_first)
    ranks = rank_order
    counts0 = [freq0[r] for r in ranks]
    counts1 = [freq1[r] for r in ranks]
    x = np.arange(len(ranks))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts0, width, label='Seats0&2 (LSTM DQN)')
    plt.bar(x + width/2, counts1, width, label='Seats1&3 (Random)')
    plt.xticks(x, ranks, rotation=45)
    plt.ylabel('Frequency')
    plt.title('First-play Card Rank Frequency by Team')
    plt.legend()
    plt.tight_layout()
    out3 = 'lstmn_first_rank_freq.png'
    plt.savefig(out3)
    print(f"Saved frequency plot to {out3}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='dqnlstm_model.pth')
    parser.add_argument('--episodes',   type=int, default=5000)
    args = parser.parse_args()
    evaluate_lstm(args.model_path, args.episodes)
