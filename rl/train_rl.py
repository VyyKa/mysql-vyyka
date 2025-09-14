from pathlib import Path
from datetime import datetime
import numpy as np, torch
from .env_log import LogEnv
from .agent import Agent

BASE     = Path(__file__).resolve().parent.parent    # project root
DATA_DIR = BASE / "data" / "cleaned"
ART_DIR  = BASE / "artifacts"
OUT_DIR  = ART_DIR / "rl"

def latest_cleaned_csv() -> Path:
    files = sorted(DATA_DIR.glob("clean_log_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files: raise FileNotFoundError("Không thấy clean_log_*.csv trong data/cleaned.")
    return files[0]

def main():
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    csv = latest_cleaned_csv()
    vec = ART_DIR / "vectorizer.joblib"
    env = LogEnv(csv, vectorizer_path=vec, bonus_weight=0.25)

    agent = Agent(
        state_dim=env.observation_dim, n_actions=env.action_space_n, out_dir=OUT_DIR,
        gamma=0.99, lr=1e-3, batch_size=256, memory_cap=200_000,
        eps_start=1.0, eps_final=0.05, eps_decay_steps=20_000
    )

    episodes = 30  # chỉnh 50 nếu muốn kỹ hơn
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Train on {csv.name} | N={env.n} | dim={env.observation_dim}")

    best_acc = 0.0
    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_reward = 0.0; correct = 0; total = 0
        done = False
        while not done:
            a = agent.act(s)
            ns, r, done, info = env.step(a)
            agent.memory.push(s, a, r, ns, done)
            agent.learn()
            s = ns
            agent.steps += 1
            total += 1; correct += int(a == info["label"]); ep_reward += r

        acc = correct / max(1, total)
        log = {"ts": datetime.now().isoformat(), "ep": ep, "episodes": episodes,
               "steps": agent.steps, "eps": round(agent.epsilon(),4),
               "ep_reward": round(ep_reward,3), "acc": round(acc,4), "best_acc": round(best_acc,4)}
        agent.log(log)
        print(f"EP {ep}/{episodes} | eps={log['eps']:.3f} | steps={agent.steps} | R={ep_reward:.1f} | acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            agent.save("dqn_best.pt")

    agent.save("dqn_last.pt")
    print(f"Saved: {OUT_DIR/'dqn_best.pt'} , {OUT_DIR/'dqn_last.pt'}")

if __name__ == "__main__":
    main()
