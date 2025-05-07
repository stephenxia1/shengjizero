import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env import Env, CARDMAP

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.vstack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.online_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, valid_actions):
    
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.choice(valid_actions)

        
        state_v = torch.FloatTensor(state).unsqueeze(0)  
        with torch.no_grad():
            q_vals = self.online_net(state_v).cpu().numpy().squeeze()  

        
        valid_qs = q_vals[valid_actions]                         
        best_idx = valid_actions[int(np.argmax(valid_qs))]       
        return best_idx


    def update(self):
        if len(self.buffer) < self.batch_size:
            # print("Buffer not enough samples to update")
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions).unsqueeze(1)
        rewards_v = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_v = torch.FloatTensor(next_states)
        dones_v = torch.ByteTensor(dones).unsqueeze(1)


        q_values = self.online_net(states_v).gather(1, actions_v)


        next_q_values = self.target_net(next_states_v).max(1)[0].unsqueeze(1)
        target_q_values = rewards_v + self.gamma * next_q_values * (1 - dones_v.float())

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

def train_dqn(
    env,
    agent,
    num_episodes=20000,
    target_update_freq=10,
    max_steps_per_episode=200
):
    episode_rewards = []
    for episode in range(1, num_episodes + 1):
        env.reset()
        state = env.reset()
        total_reward = 0
        # print("Reset environment.")
        for t in range(max_steps_per_episode):
            while (env.current_player != 0 and not env.done):
                # print(f'{env.current_player} is playing')
                env.step(np.random.choice(np.where(env.getValidActions(env.current_player)==1)[0]))
            
            state = env.getObs(env.current_player)
            # print(f'0 is playing')
            action = agent.select_action(state, np.where(env.getValidActions(env.current_player)==1)[0])
            env.step(action)

            while (env.current_player != 0 and not env.done):
                env.step(np.random.choice(np.where(env.getValidActions(env.current_player)==1)[0]))

            reward = env.get_reward(0)
            done = env.done
            next_state = env.getObs(0)

            agent.buffer.push(state, action, reward, next_state, done)

            agent.update()
            state = next_state
            total_reward += reward
            if done:
                # print(f'Game {episode} finished where attacking team scored {env.points} points, reward: {reward}, landlord team:{env.landlord % 2 == 0}')
                break

        episode_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target()

        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode}/{num_episodes}, Average Reward: {avg_reward:.2f}")

    return episode_rewards

def evaluate_agent(env, agent, num_episodes=1000):
    total_rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        done = False
        while not done:
            while (env.current_player != 0 and not env.done):
                action = np.random.choice(np.where(env.getValidActions(env.current_player)==1)[0])
                if episode == 0:
                    print(f'{env.current_player} is playing {CARDMAP[action]}, HAND: {[CARDMAP[j] for j in np.where(env.hands[env.current_player]==1)[0]]}')
                env.step(action)
            action = agent.select_action(env.getObs(0), np.where(env.getValidActions(env.current_player)==1)[0])
            if episode == 0:
                print(f'0 is playing {CARDMAP[action]}, HAND: {[CARDMAP[j] for j in np.where(env.hands[env.current_player]==1)[0]]}')
            env.step(action)
            while (env.current_player != 0 and not env.done):
                action = np.random.choice(np.where(env.getValidActions(env.current_player)==1)[0])
                if episode == 0:
                    print(f'{env.current_player} is playing {CARDMAP[action]}, HAND: {[CARDMAP[j] for j in np.where(env.hands[env.current_player]==1)[0]]}')
                env.step(action)

            reward = env.get_reward(0)
            done = env.done
            state = env.getObs(0)
            total_reward += reward
        total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    env = Env() 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DQNAgent(state_dim, action_dim)
    rewards = train_dqn(env, agent)

    evaluate_agent(env, agent)
    # Save the model
    torch.save(agent.online_net.state_dict(), "dqn_model.pth")