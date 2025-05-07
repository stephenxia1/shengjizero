import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from dqn import *
import env


def train_self_play(env,
                    agents,
                    num_episodes: int = 1000,
                    seq_len: int = 4,
                    target_update: int = 100,
                    warmup_steps: int = 1000):
    
    episode_rewards = [[], [], [], []]
    num_agents = len(agents)
    for episode in range(1, num_episodes + warmup_steps + 1):
        env.reset()
        while not env.done:
            
            player = env.current_player
            agent = agents[player]
            
            eps = 0.05 + (0.9 - 0.05) * math.exp(-1. * agent.steps_done/ 200)

            state = env.getObs(player)
            if episode > warmup_steps:
                action = agent.select_action(state, [j for j in np.where(env.getValidActions(player)==1)[0]])
            else:
                action = np.random.choice(np.where(env.getValidActions(player)==1)[0])
            env.step(action)
            next_state = env.getObs(player)
            reward = env.get_reward(player)            
            agent.buffer.push(state, action, reward, next_state, env.done)
            if episode > warmup_steps:
                agent.update()
            agent.steps_done += 1
        
        for player in range(num_agents):
            episode_rewards[player].append(env.get_reward(player))

            if episode % target_update == 0:
                agent.update_target()

            if episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[player][-20:])
                print(f"Episode {episode}/{num_episodes}, Agent {player} Average Reward: {avg_reward:.2f}")

def evaluate_self_play(env, agents, num_episodes: int = 1000):

    total_rewards = [[], [], [], []]
    num_agents = len(agents)
    for episode in range(num_episodes):
        env.reset()
        while not env.done:
            player = env.current_player
            agent = agents[player]
            state = env.getObs(player)
            action = agent.select_action(state, [j for j in np.where(env.getValidActions(player)==1)[0]])
            env.step(action)
            if (episode == 0):
                values = agent.online_net(torch.FloatTensor(state).unsqueeze(0)).detach().numpy().squeeze()
                cardvalues = {CARDMAP[j] : float(values[j]) for j in range(len(values))}
                print(f'Player {env.current_player} action values: {cardvalues}')
                print(f'{env.current_player} is playing {CARDMAP[action]}, HAND: {[CARDMAP[j] for j in np.where(env.hands[env.current_player]==1)[0]]}')
        
        for player in range(num_agents):
            total_rewards[player].append(env.get_reward(player))

    for player in range(num_agents):
        avg_reward = np.mean(total_rewards[player])
        print(f"Agent {player} Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    env = Env()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 128
    num_agents = 4

    agents = [DQNAgent(state_dim, action_dim) for _ in range(num_agents)]

    train_self_play(env, agents, num_episodes=10000)
    for i in range(len(agents)):
        torch.save(agents[i].online_net.state_dict(), f"agent{i}.pth")

    evaluate_self_play(env, agents, num_episodes=1000)

