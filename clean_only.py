# clean_only.py
# Đọc parsed_log_*.csv -> chuẩn hóa -> tính reward/risky -> save clean_log_*.csv (có log rõ ràng)

import os, re, glob
import pandas as pd

PARSED_DIR = "data/parsed"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)

# ---- RULES ----
def compute_reward(q: str) -> int:
    s = q.lower()
    # cực nguy hiểm
    if re.search(r"\bdrop\s+(database|table)\b", s): return -3
    if re.search(r"\btruncate\b", s): return -2
    if re.search(r"\bdelete\b(?!.*\bwhere\b)", s): return -2
    if re.search(r"\bupdate\b(?!.*\bwhere\b)", s): return -2
    # quyền & user
    if re.search(r"\b(grant|revoke|create\s+user|alter\s+user)\b", s): return -1
    # an toàn / thường gặp
    if re.search(r"\b(select|insert|show|describe|explain)\b", s): return +1
    return 0

def normalize_sql(s: str) -> str:
    if not isinstance(s, str): s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"'[^']*'", "'str'", s)
    s = re.sub(r'"[^"]*"', '"str"', s)
    s = re.sub(r'\b\d+\b', '0', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def pick_latest_parsed():
    paths = sorted(glob.glob(os.path.join(PARSED_DIR, "parsed_log_*.csv")))
    if not paths:
        raise FileNotFoundError(f"Không thấy file parsed trong {PARSED_DIR}. Hãy chạy parse_log.py trước.")
    return paths[-1]

def main():
    in_path = pick_latest_parsed()
    print(f"📥 Load parsed: {in_path}")
    df = pd.read_csv(in_path)

    # tìm cột chứa SQL
    sql_col = None
    for c in ["normalized_query","sql","query","statement","command","text"]:
        if c in df.columns:
            sql_col = c; break
    if sql_col is None:
        raise KeyError("Không tìm thấy cột SQL (normalized_query/sql/query/statement/command).")

    # chuẩn hoá
    if "normalized_query" not in df.columns:
        df["normalized_query"] = df[sql_col].astype(str).map(normalize_sql)
    else:
        df["normalized_query"] = df["normalized_query"].astype(str).map(normalize_sql)

    # query_type đơn giản
    def get_type(s:str)->str:
        m = re.match(r'^\s*(\w+)', s)
        return (m.group(1).upper() if m else "UNK")
    df["query_type"] = df["normalized_query"].map(get_type)

    # reward & risky
    df["reward"] = df["normalized_query"].map(compute_reward)
    df["risky"]  = (df["reward"] < 0).astype("int64")

    # chọn cột gọn gàng
    keep = ["normalized_query","query_type","reward","risky"]
    extra = [c for c in ["user","host","db","ts","timestamp"] if c in df.columns]
    out_df = df[keep + extra]

    # lưu
    idx = 1 + len(glob.glob(os.path.join(CLEAN_DIR, "clean_log_*.csv")))
    out_path = os.path.join(CLEAN_DIR, f"clean_log_{idx}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved -> {out_path} (rows={len(out_df):,})")
    print("   Columns:", ", ".join(out_df.columns))

if __name__ == "__main__":
    main()
