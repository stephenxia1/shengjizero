
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from jointqn import JointQNetwork
from env1    import Env1
from collections import Counter


def evaluate_joint_dqn(model_path, episodes=1000):
    env = Env1()
    state_dim  = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
    policy = JointQNetwork(state_dim, hidden_dim=256, action_dim=action_dim)
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()

    team0_returns = []  
    team1_returns = []  
    flags = []      
    detailed = []       
    team0_first = []     
    team1_first = []     

    init_hand_pts_02 = [] 
    earned_pts_02   = []
    init_hand_pts_13 = []  
    earned_pts_13   = []
    first02 = []        
    first13 = []       

    for ep in range(1, episodes+1):
        env.reset()
        obs0 = env.getObs(0)
        obs2 = env.getObs(2)
        flag = (env.landlord % 2)
        flags.append(flag)
        done = False
        first_card = None
        if flag == 0:
            pts_init = env.getPoints(env.hands[1]) + env.getPoints(env.hands[3])
            init_hand_pts_13.append(pts_init)
        else:
            pts_init = env.getPoints(env.hands[0]) + env.getPoints(env.hands[2])
            init_hand_pts_02.append(pts_init)
        while not done:
            new_trick = (env.current_trick.sum() == 0)
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
                a0, a2 = np.unravel_index(np.argmax(qjoint), qjoint.shape)
                action = a0 if player==0 else a2
                if new_trick and first_card is None:
                    first_card = action
            else:
                valid = env.getValidActions(player)
                opts  = np.where(valid==1)[0]
                action = int(random.choice(opts)) if opts.size else 0
                if new_trick and first_card is None:
                    first_card = action

            cur_valid = env.getValidActions(player)
            if not cur_valid[action]:
                choices = np.where(cur_valid==1)[0]
                action = int(random.choice(choices)) if choices.size else 0

            env.step(action)
            done = env.done
            obs0 = env.getObs(0)
            obs2 = env.getObs(2)

        r0 = env.get_reward(0)
        r1 = env.get_reward(1)
        r2 = env.get_reward(2)
        r3 = env.get_reward(3)
        if flag == 0:
            team1_returns.append(r1)
            detailed.append(r1)
            team0_first.append(first_card)
        else:
            team0_returns.append(r0)
            detailed.append(r0)
            team1_first.append(first_card)

        print(f"Episode {ep}/{episodes}: team return = {detailed[-1]:.2f}")

        if first_card is not None:
            (first02 if flag==0 else first13).append(first_card)

        raw_pts = env.points
        if flag == 0:
            earned_pts_13.append(raw_pts)
        else:
            earned_pts_02.append(raw_pts)

    mean0 = np.mean(team0_returns) if team0_returns else 0.0
    std0  = np.std(team0_returns)  if team0_returns else 0.0
    mean1 = np.mean(team1_returns) if team1_returns else 0.0
    std1  = np.std(team1_returns)  if team1_returns else 0.0

    print(f"\nEvaluation over {episodes} episodes:")
    print(f"Seats 1&3 (landlord eps): mean={mean0:.2f}, std={std0:.2f}, count={len(team0_returns)}")
    print(f"Seats 0&2 (landlord eps): mean={mean1:.2f}, std={std1:.2f}, count={len(team1_returns)}")

    labels = ['Seats 0&2 (Joint DQN)', 'Seats 1&3 (Random)']
    means  = [mean0, mean1]
    errs   = [std0, std1]
    x = np.arange(2)
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Mean Team Return')
    plt.title('JointDQN Team Performance (Landlord Episodes)')
    plt.tight_layout()
    for xi, (mu, sd) in enumerate(zip(means, errs)):
        plt.text(xi, mu + sd + 0.02, f"{mu:.2f}±{sd:.2f}",
                 ha='center', va='bottom', fontsize=10)
    plt.savefig('joint_team_bar_filtered.png')
    print("Saved filtered team bar chart.")
    plt.show()


    rank_order = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','Joker']
    def rank_label(idx):
        if idx>=52: return 'Joker'
        if idx>=48: return '2'
        r = idx % 12
        return {11:'A',10:'K',9:'Q',8:'J',7:'10'}.get(r, str(r+3))

    freq0 = Counter(rank_label(c) for c in team0_first if c is not None)
    freq1 = Counter(rank_label(c) for c in team1_first if c is not None)
    counts0 = [freq0.get(r,0) for r in rank_order]
    counts1 = [freq1.get(r,0) for r in rank_order]

    width = 0.35
    x = np.arange(len(rank_order))
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts0, width, label='Joint DQN')
    plt.bar(x + width/2, counts1, width, label='Random')
    plt.xticks(x, rank_order, rotation=45)
    plt.ylabel('Frequency')
    plt.title('First-play Card Rank Frequency: Joint DQN vs Random')
    plt.legend()
    plt.tight_layout()
    plt.savefig('joint_team_first_rank_freq.png')
    print("Saved frequency plot.")
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
    plt.xticks(x_pts,['Joint DQN', 'Random'])
    plt.ylabel('Points')
    plt.title('Points in Hand vs Points Earned: Joint DQN vs Random')
    for b,var in zip(bars_i, ['Init Pts','Init Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    for b,var in zip(bars_e, ['End Pts','End Pts']):
        h=b.get_height(); x0=b.get_x()+b.get_width()/2
        plt.text(x0,h+1,f'{var}={h:.1f}',ha='center',va='bottom')
    plt.legend(); plt.tight_layout(); plt.savefig('jointdqn_points_compare.png'); plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='joint_team_dqn.pth')
    parser.add_argument('--episodes',   type=int, default=5000)
    args = parser.parse_args()
    evaluate_joint_dqn(args.model_path, args.episodes)
