import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque
from env import Env, CARDMAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared model definitions
class DQNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=1):
        super(DQNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        q = self.fc(out[:, -1, :])
        return q, hidden

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# Simple replay buffer for learner
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        # ignore None transitions
        if transition is None:
            return
        self.buffer.append(transition)

    def sample(self, batch_size):
        # filter out any None if present
        valid_buf = [t for t in self.buffer if t is not None]
        batch = random.sample(valid_buf, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

# Actor process: collects experiences and pushes to queue
def actor_loop(pid, policy_state_dict, exp_queue, stop_flag, warmup_steps, eps_start, eps_end, eps_decay):
    torch.manual_seed(pid)
    env = Env()
    model = DQNLSTM(env.observation_space.shape[0], 128, env.action_space.shape[0]).to(device)
    model.load_state_dict(policy_state_dict)
    model.eval()

    total_steps = 0
    hidden = model.init_hidden()
    state = env.reset()

    while not stop_flag.value:
        player = env.current_player
        valid = env.getValidActions(player)
        state = env.getObs(player)
        eps = eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay)
        if random.random() < eps:
            action = int(np.random.choice(np.where(valid == 1)[0]))
        else:
            s = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                q, hidden = model(s, hidden)
            q = q.cpu().numpy().flatten()
            q[valid == 0] = -np.inf
            action = int(np.argmax(q))

        next_state = env.step(action)
        reward = env.get_reward(player)
        done = env.done
        next_hidden = hidden

        # push experience, skip None
        exp_queue.put((state, hidden, action, reward, next_state, next_hidden, done))

        state = next_state
        total_steps += 1
        if done:
            state = env.reset()
            hidden = model.init_hidden()
    # signal termination for each actor
    exp_queue.put(None)

# Learner: reads from queue, updates model
def learner_loop(num_actors, policy_net, target_net, exp_queue, stop_flag,
                 buffer_size, batch_size, gamma, target_update, warmup_steps):
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay = ReplayBuffer(buffer_size)
    steps = 0
    finished_actors = 0

    while True:
        item = exp_queue.get()
        if item is None:
            finished_actors += 1
            if finished_actors == num_actors:
                break
            continue

        state, hidden, action, reward, next_state, next_hidden, done = item
        # store only valid transitions
        replay.push((state, hidden, action, reward, next_state, next_hidden, done))

        # optimize if enough samples
        if len(replay) > batch_size and steps > warmup_steps:
            states, hiddens, actions, rewards, next_states, next_hiddens, dones = replay.sample(batch_size)
            sb = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device)
            nsb = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(device)
            ab = torch.LongTensor(actions).unsqueeze(1).to(device)
            rb = torch.FloatTensor(rewards).to(device)
            db = torch.FloatTensor(dones).to(device)

            h0 = torch.cat([h[0] for h in hiddens], dim=1).to(device)
            c0 = torch.cat([h[1] for h in hiddens], dim=1).to(device)
            nh0 = torch.cat([h[0] for h in next_hiddens], dim=1).to(device)
            nc0 = torch.cat([h[1] for h in next_hiddens], dim=1).to(device)

            qvals, _ = policy_net(sb, (h0, c0))
            state_act = qvals.gather(1, ab).squeeze(1)

            with torch.no_grad():
                qnext, _ = target_net(nsb, (nh0, nc0))
                max_next = qnext.max(1)[0]
                expected = rb + (1 - db) * gamma * max_next

            loss = nn.MSELoss()(state_act, expected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
        steps += 1

        if steps % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Target network updated at step {steps}")

    stop_flag.value = 1

if __name__ == '__main__':
    mp.set_start_method('spawn')
    num_actors = 4
    buffer_size = 50000
    batch_size = 64
    gamma = 0.99
    target_update = 1000
    warmup_steps = 1000
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 50000

    temp_env = Env()
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.shape[0]

    policy_net = DQNLSTM(obs_dim, 128, act_dim)
    target_net = DQNLSTM(obs_dim, 128, act_dim)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.share_memory()
    target_net.share_memory()

    exp_queue = mp.Queue(maxsize=buffer_size)
    stop_flag = mp.Value('i', 0)
    actors = []
    policy_state = policy_net.state_dict()
    for i in range(num_actors):
        p = mp.Process(target=actor_loop, args=(i, policy_state, exp_queue,
                                                stop_flag, warmup_steps,
                                                eps_start, eps_end, eps_decay))
        p.start()
        actors.append(p)

    learner_loop(num_actors, policy_net, target_net, exp_queue,
                 stop_flag, buffer_size, batch_size, gamma, target_update, warmup_steps)

    for p in actors:
        p.join()

    torch.save(policy_net.state_dict(), 'policy_parallel.pth')
    print('Training complete. Model saved as policy_parallel.pth')
