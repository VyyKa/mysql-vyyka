import json, random, numpy as np, torch
from collections import deque, namedtuple
from pathlib import Path
from .model import DQN

Transition = namedtuple("Transition", "state action reward next_state done")

def to_tensor(x, device=None, dtype=torch.float32):
    if isinstance(x, np.ndarray): t = torch.from_numpy(x).to(dtype)
    else: t = torch.tensor(x, dtype=dtype)
    return t.to(device) if device is not None else t

class ReplayMemory:
    def __init__(self, cap=100_000): self.buffer = deque(maxlen=cap)
    def push(self, *a): self.buffer.append(Transition(*a))
    def sample(self, batch): 
        B = random.sample(self.buffer, batch)
        s = np.stack([b.state for b in B]); a = np.array([b.action for b in B], np.int64)
        r = np.array([b.reward for b in B], np.float32)
        ns = np.stack([b.next_state for b in B]); d = np.array([b.done for b in B], np.int64)
        return s,a,r,ns,d
    def __len__(self): return len(self.buffer)

class Agent:
    def __init__(self, state_dim, n_actions, out_dir: Path,
                 gamma=0.99, lr=1e-3, batch_size=256, memory_cap=200_000,
                 eps_start=1.0, eps_final=0.05, eps_decay_steps=20_000, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, n_actions, hidden_size=512, gamma=gamma, lr=lr, device=self.device)
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_cap)
        self.steps = 0
        self.eps_start, self.eps_final, self.eps_decay = eps_start, eps_final, eps_decay_steps
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.log_f = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")

    def epsilon(self):
        frac = min(1.0, self.steps / max(1, self.eps_decay))
        return self.eps_final + (self.eps_start - self.eps_final) * (1.0 - frac)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.model.policy_net.advantage[-1].out_features)
        s = to_tensor(state, self.device).unsqueeze(0)
        return int(self.model.act(s).item())

    def learn(self):
        if len(self.memory) < self.batch_size: return None
        s,a,r,ns,d = self.memory.sample(self.batch_size)
        s  = to_tensor(s,  self.device)
        a  = to_tensor(a,  self.device, torch.long)
        r  = to_tensor(r,  self.device)
        ns = to_tensor(ns, self.device)
        d  = to_tensor(d,  self.device, torch.long)
        return self.model.update(s,a,r,ns,d)

    def save(self, name): torch.save(self.model.policy_net.state_dict(), self.out_dir / name)
    def log(self, payload: dict): self.log_f.write(json.dumps(payload, ensure_ascii=False)+"\n"); self.log_f.flush()
