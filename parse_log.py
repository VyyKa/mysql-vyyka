import re, os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_logs"
OUT_DIR = BASE_DIR / "data" / "parsed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def next_name():
    files = list(OUT_DIR.glob("parsed_log_*.csv"))
    nums = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
    n = max(nums, default=0) + 1
    return OUT_DIR / f"parsed_log_{n}.csv"

def main():
    # lấy file log mới nhất trong raw_logs
    logs = sorted(RAW_DIR.glob("*.log"), key=os.path.getmtime)
    if not logs:
        print("❗ Không tìm thấy file .log trong data/raw_logs/")
        return
    in_file = logs[-1]
    print(f"Đang parse: {in_file.name}")

    rows = []
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2}[^ ]+).*?\s(Query|Execute)\s+(.*)$")
    with open(in_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = ts_re.match(line)
            if m:
                ts = m.group(1)
                q = m.group(3).strip()
                rows.append({"time": ts, "query": q})

    df = pd.DataFrame(rows)
    out = next_name()
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Saved -> {out} (rows={len(df)})")

if __name__ == "__main__":
    main()
