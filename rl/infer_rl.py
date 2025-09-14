# rl/infer_rl.py
# Inference 1 câu SQL với DuelingQNet + rule override, tự rebuild SVD khi mismatch

import os
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
import torch.nn as nn

from rl.env_log import load_vectorizer, load_or_fit_svd, transform_query_for_rl, normalize_sql

# ----- Model -----
class DuelingQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

# ----- Rule override đơn giản -----
import re
RULE_PATTERNS = [
    re.compile(r"\bdrop\s+(database|table)\b", re.I),
    re.compile(r"\btruncate\b", re.I),
    re.compile(r"\bdelete\b(?!.*\bwhere\b)", re.I),
    re.compile(r"\bupdate\b(?!.*\bwhere\b)", re.I),
    re.compile(r"\b(grant|revoke|create\s+user|alter\s+user)\b", re.I),
]

def is_rule_risky(sql: str) -> bool:
    s = sql if isinstance(sql, str) else ""
    return any(p.search(s) for p in RULE_PATTERNS)

# ----- Paths -----
ART_DIR = "artifacts"
RL_DIR  = os.path.join(ART_DIR, "rl")
os.makedirs(RL_DIR, exist_ok=True)

def build_net(input_dim: int, n_actions: int = 2, hidden: int = 512, device="cpu"):
    net = DuelingQNet(input_size=input_dim, hidden_size=hidden, output_size=n_actions).to(device)
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sql", type=str, help="Câu SQL cần dự đoán")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load vectorizer
    vec = load_vectorizer()

    # 2) Load / auto-fit SVD (rebuild nếu mismatch)
    svd = load_or_fit_svd(vec, n_components=512, random_state=42)
    input_dim = getattr(svd, "n_components", 512)

    # 3) Build net & load weights (lenient)
    dqn_path = os.path.join(RL_DIR, "dqn_best.pt")
    if not os.path.exists(dqn_path):
        # fallback tên khác
        dqn_path = os.path.join(RL_DIR, "dqn_last.pt")
        if not os.path.exists(dqn_path):
            # nếu chưa train RL, tạo model “rỗng” để dự đoán giả lập (Q gần bằng 0)
            print("⚠️  Không tìm thấy dqn_*.pt, sẽ dùng mạng khởi tạo ngẫu nhiên.")
            dqn = build_net(input_dim, device=device)
        else:
            dqn = build_net(input_dim, device=device)
            state = torch.load(dqn_path, map_location=device, weights_only=True)
            dqn.load_state_dict(state, strict=False)
    else:
        dqn = build_net(input_dim, device=device)
        state = torch.load(dqn_path, map_location=device, weights_only=True)
        dqn.load_state_dict(state, strict=False)

    dqn.eval()

    # 4) Transform câu SQL
    x_np = transform_query_for_rl(args.sql, vec, svd)  # (1, k)
    x = torch.from_numpy(x_np).to(device)

    with torch.no_grad():
        q = dqn(x).cpu().numpy().ravel()
    label_idx = int(np.argmax(q))
    label = "benign" if label_idx == 0 else "risky"

    # 5) Rule override
    rule_hit = False
    if is_rule_risky(normalize_sql(args.sql)):
        label = "risky"
        rule_hit = True

    # 6) In ra console
    q_vals = [float(f"{v:.3f}") for v in q.tolist()]
    print(f"SQL: {args.sql}")
    print(f"Predicted: {label}{' (rule override)' if rule_hit else ''}")
    print(f"Q-values: {q_vals}")

    # 7) Ghi JSONL để ELK ingest
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "sql": args.sql,
        "label": label,
        "q_values": q.tolist(),
        "reward_est": float(np.tanh(q[1] - q[0])),  # ước lượng đơn giản từ chênh lệch Q
        "rule_hit": rule_hit,
    }
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "rl_pred.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    main()
