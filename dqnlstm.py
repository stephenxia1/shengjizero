import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# Assume Env and CARDMAP are defined in env.py in the same directory
from env import Env, CARDMAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=2):
        super(DQNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        # x: (batch, seq_len=1, input_dim)
        out, hidden = self.lstm(x, hidden)
        q = self.fc(out[:, -1, :])  # use last output
        return q, hidden

    def init_hidden(self, batch_size=1):
        # initialize hidden and cell states to zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, hidden, action, reward, next_state, next_hidden, done):
        self.buffer.append((state, hidden, action, reward, next_state, next_hidden, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, hidden, action, reward, next_state, next_hidden, done = zip(*batch)
        return state, hidden, action, reward, next_state, next_hidden, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim=128, lr=1e-3,
                 gamma=0.999, buffer_size=20000, batch_size=64, target_update=500):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DQNLSTM(obs_dim, hidden_dim, action_dim).to(device)
        self.target_net = DQNLSTM(obs_dim, hidden_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state, hidden, valid_actions, eps):
        # state: numpy array; hidden: (h,c) tensors
        if random.random() < eps:
            # random valid action
            valid_idxs = np.where(valid_actions == 1)[0]
            action = random.choice(valid_idxs)
            return action, hidden
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,obs_dim)
                q_values, hidden = self.policy_net(s, hidden)
                q_values = q_values.cpu().numpy().flatten()
                # mask invalid
                q_values[valid_actions == 0] = -np.inf
                action = int(np.argmax(q_values))
                return action, hidden

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        states, hiddens, actions, rewards, next_states, next_hiddens, dones = self.memory.sample(self.batch_size)

        # Prepare tensors
        state_batch = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device)        # (B,1,obs_dim)
        next_state_batch = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        done_batch = torch.FloatTensor(dones).to(device)

        # Hidden states for batch (stack h and c)
        h_batch = torch.cat([h[0] for h in hiddens], dim=1).to(device)
        c_batch = torch.cat([h[1] for h in hiddens], dim=1).to(device)
        hidden_batch = (h_batch, c_batch)

        nh_batch = torch.cat([h[0] for h in next_hiddens], dim=1).to(device)
        nc_batch = torch.cat([h[1] for h in next_hiddens], dim=1).to(device)
        next_hidden_batch = (nh_batch, nc_batch)

        # Compute Q(s,a)
        q_values, _ = self.policy_net(state_batch, hidden_batch)
        state_action_values = q_values.gather(1, action_batch).squeeze(1)

        # Compute V(s')
        with torch.no_grad():
            q_next, _ = self.target_net(next_state_batch, next_hidden_batch)
            max_next = q_next.max(1)[0]
            expected_values = reward_batch + (1 - done_batch) * self.gamma * max_next

        # Loss
        loss = nn.MSELoss()(state_action_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training loop
if __name__ == "__main__":
    num_agents = 4
    env = Env()
    obs_dim = env.observation_space.shape[0]
    action_dim = 54

    agents = [DQNAgent(obs_dim, action_dim) for _ in range(num_agents)]

    num_episodes = 70000
    max_steps = 1000
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 50000  # decay over steps
    warmup_steps = 20000
    total_steps = 0

    for ep in tqdm(range(1, num_episodes+1)):
        state = env.reset()
        # initial hidden states for each agent
        hiddens = [agent.policy_net.init_hidden() for agent in agents]

        for t in range(max_steps):
            player = env.current_player
            agent = agents[player]

            valid = env.getValidActions(player)
            eps = eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay)
            state = env.getObs(player)
            action, new_hidden = agent.select_action(state, hiddens[player], valid, eps)
            next_state = env.step(action)
            reward = env.get_reward(player)
            done = env.done

            # update hidden states
            next_hiddens = hiddens.copy()
            next_hiddens[player] = new_hidden

            # store transition
            agent.memory.push(state, hiddens[player], action, reward, next_state, new_hidden, done)
            if total_steps > warmup_steps:
                agent.optimize_model()

            hiddens = next_hiddens
            total_steps += 1

            if done:
                break

        # if ep % 100 == 0:
            # print(f"Episode {ep}/{num_episodes}, Total Steps: {total_steps}, Epsilon: {eps:.2f}")
        # update target networks periodically
        for agent in agents:
            if ep % agent.target_update == 0 and total_steps > warmup_steps:
                agent.update_target()

    # Save trained models
    for idx, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f"agent_{idx}.pth")

    # Evaluation
    num_eval = 50
    rewards_sum = np.zeros(num_agents)
    for _ in range(num_eval):
        state = env.reset()
        hiddens = [agent.policy_net.init_hidden() for agent in agents]
        while not env.done:
            player = env.current_player
            agent = agents[player]
            valid = env.getValidActions(player)
            state = env.getObs(player)
            action, new_hidden = agent.select_action(state, hiddens[player], valid, eps=0.0)
            state = env.step(action)
            hiddens[player] = new_hidden
            rewards_sum[player] += env.get_reward(player)
    avg_rewards = rewards_sum / num_eval
    print("Average rewards over evaluation episodes:", avg_rewards)

    # Print one round of gameplay
    state = env.reset()
    hiddens = [agent.policy_net.init_hidden() for agent in agents]
    print("--- One round of gameplay ---")
    while not env.done:
        player = env.current_player
        agent = agents[player]
        valid = env.getValidActions(player)
        state = env.getObs(player)
        action, new_hidden = agent.select_action(state, hiddens[player], valid, eps=0.0)
        print(f"Player {player} plays {CARDMAP[action]},HAND: {[CARDMAP[j] for j in np.where(env.hands[player]==1)[0]]}")
        state = env.step(action)
        hiddens[player] = new_hidden
    print(f"Final Points: {env.points}")
    for i in range(num_agents):
        print(f"Player {i} reward: {env.get_reward(i)}")
