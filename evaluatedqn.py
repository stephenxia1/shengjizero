# Regular DQN vs Random
import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

from dqn import QNetwork
from env1 import Env1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dqn(model_path, episodes=100):
    env = Env1()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found.")
    model = QNetwork(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    
    team0_returns = []   
    team1_returns = []    
    flags         = []   
    first02       = []    
    first13       = []    

   
    init_hand_pts_02, earned_pts_02 = [], []
    init_hand_pts_13, earned_pts_13 = [], []

    for ep in range(1, episodes+1):
        obs = env.reset()
        landlord_flag = env.landlord % 2
        flags.append(landlord_flag)

       
        if landlord_flag == 0:
            init_hand_pts_13.append(env.getPoints(env.hands[1]) + env.getPoints(env.hands[3]))
        else:
            init_hand_pts_02.append(env.getPoints(env.hands[0]) + env.getPoints(env.hands[2]))

        first_card = None
        done = False

        while not done:
            player = env.current_player


            if player in (0, 2):
                vec = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = model(vec).cpu().numpy().flatten()

                valid = env.getValidActions(player).astype(bool)
                q[~valid] = -np.inf
                action = int(np.argmax(q))
            else:
                valid = env.getValidActions(player)
                opts  = np.where(valid==1)[0]
                action = int(random.choice(opts)) if opts.size else 0

       
            if first_card is None and player == env.landlord:
                first_card = action

            obs, _, done, _ = env.step(action)

        
        r = [env.get_reward(p) for p in range(4)]
        if landlord_flag == 0:
            team1_returns.append(r[1])
            first02.append(first_card)
            earned_pts_13.append(env.points)
        else:
            team0_returns.append(r[0])
            first13.append(first_card)
            earned_pts_02.append(env.points)

        print(f"Ep {ep}/{episodes}: flag={landlord_flag}, return={r[landlord_flag]}, first_card={first_card}")

 
    m0, s0 = (np.mean(team0_returns), np.std(team0_returns)) if team0_returns else (0,0)
    m1, s1 = (np.mean(team1_returns), np.std(team1_returns)) if team1_returns else (0,0)
    print(f"Seats0&2 landlord: mean={m0:.2f}, std={s0:.2f}, count={len(team0_returns)}")
    print(f"Seats1&3 landlord: mean={m1:.2f}, std={s1:.2f}, count={len(team1_returns)}")


    labels = ['Seats 0 & 2 (DQN)', 'Seats 1 & 3 (Random)']
    means = [m0, m1]
    errs = [s0, s1]
    
    x = np.arange(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Team Return')
    plt.title('Mean Returns (Playing Against Landlord): DQN vs Random')
    plt.tight_layout()
    for xi, (mu, sd) in enumerate(zip(means, errs)):
        plt.text(xi, mu + sd + 0.02, f"{mu:.2f}±{sd:.2f}",
                 ha='center', va='bottom', fontsize=10)
    out_bar = 'dqn_mean_returns.png'
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
    plt.xticks(x_pts,['DQN','Random'])
    plt.ylabel('Points')
    plt.title('Points in Hand vs Points Earned: DQN vs Random')

    for b,var in zip(bars_i, ['Init Pts','Init Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    for b,var in zip(bars_e, ['End Pts','End Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    plt.legend(); plt.tight_layout(); plt.savefig('dqn_points_compare.png'); plt.show()



    rank_order = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','Joker']
    def rank_label(idx):
        if idx>=52: return 'Joker'
        if idx>=48: return '2'
        r=idx%12
        return {11:'A',10:'K',9:'Q',8:'J',7:'10'}.get(r,str(r+3))

    freq0 = Counter(rank_label(c) for c in first02)
    freq1 = Counter(rank_label(c) for c in first13)
    ranks = rank_order
    counts0 = [freq0[r] for r in ranks]
    counts1 = [freq1[r] for r in ranks]
    x = np.arange(len(ranks))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts0, width, label='Seats0&2 (DQN)')
    plt.bar(x + width/2, counts1, width, label='Seats1&3 (Random)')
    plt.xticks(x, ranks, rotation=45)
    plt.ylabel('Frequency')
    plt.title('First-play Card Rank Frequency by Team: DQN vs Random')
    plt.legend()
    plt.tight_layout()
    out3 = 'dqn_first_rank_freq.png'
    plt.savefig(out3)
    print(f"Saved frequency plot to {out3}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='dqn_model.pth')
    parser.add_argument('--episodes',   type=int, default=5000)
    args = parser.parse_args()
    evaluate_dqn(args.model_path, args.episodes)
