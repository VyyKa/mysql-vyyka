import torch, torch.nn as nn, torch.optim as optim

class DuelingQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)

class DQN:
    """Dueling DQN + DoubleDQN + Soft Target Update"""
    def __init__(self, input_size, output_size, hidden_size=512, gamma=0.99, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.policy_net = DuelingQNet(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DuelingQNet(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, states):  # states: [B, dim]
        with torch.no_grad():
            q = self.policy_net(states)
            return q.argmax(dim=1)

    def update(self, states, actions, rewards, next_states, dones, tau=0.005):
        q_val = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q
        loss = self.loss_fn(q_val, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.opt.step()
        # soft update
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1 - tau).add_(pp.data * tau)
        return float(loss.item())
