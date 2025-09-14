# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import datetime, json
from pathlib import Path
import numpy as np
import torch

from rl.model import DuelingDQN          # giữ nguyên như project của bạn
from rl.env_log import (
    load_vectorizer_and_svd,
    transform_query_for_rl,
)

LOG_FILE = Path("logs/rl_pred.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_prediction(sql: str, label: str, q_values: list[float], reward_est: float):
    rec = {
        "ts": datetime.datetime.now().isoformat(),
        "sql": sql,
        "label": label,
        "q_values": q_values,
        "reward_est": reward_est,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

app = FastAPI(title="DBS RL Inference API")

class Req(BaseModel):
    sql: str

class Resp(BaseModel):
    label: str
    q_values: list[float]
    reward_est: float

# ===== Lazy load artifacts/model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VEC, SVD = load_vectorizer_and_svd(
    "artifacts/vectorizer.joblib",
    "artifacts/rl/svd_512.joblib"
)
DQN = DuelingDQN(input_dim=512, n_actions=2).to(device)
state = torch.load("artifacts/rl/dqn_best.pt", map_location=device, weights_only=True)
DQN.load_state_dict(state)
DQN.eval()

@app.post("/infer", response_model=Resp)
def infer(req: Req):
    x_np: np.ndarray = transform_query_for_rl(req.sql, VEC, SVD)  # shape (512,)
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        q = DQN(x)[0]
        qv = q.tolist()
        label = "risky" if qv[1] >= qv[0] else "benign"
        # Độ tự tin quy về [-1, 1]
        conf = abs(qv[1] - qv[0]) / (abs(qv[1]) + abs(qv[0]) + 1e-6)
        reward_est = 2 * conf - 1.0

    # Ghi ra JSONL để Logstash đọc
    log_prediction(req.sql, label, qv, reward_est)
    return Resp(label=label, q_values=qv, reward_est=reward_est)
