# rl/env_log.py
# Utilities cho RL: load vectorizer, hstack TF-IDF, SVD 512 & transform query

import os, glob, re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD

ART_DIR = "artifacts"
RL_DIR  = os.path.join(ART_DIR, "rl")
DATA_CLEAN = "data/cleaned"
os.makedirs(RL_DIR, exist_ok=True)

# --------- Normalize tối thiểu giống pipeline ----------
_norm_num = re.compile(r"\b\d+\b")
_norm_sgl = re.compile(r"'[^']*'")
_norm_dbl = re.compile(r'"[^"]*"')
_norm_ws  = re.compile(r"\s+")

def normalize_sql(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = _norm_sgl.sub("'str'", s)
    s = _norm_dbl.sub('"str"', s)
    s = _norm_num.sub("0", s)
    s = _norm_ws.sub(" ", s).strip()
    return s

# --------- Vectorizer helpers ----------
def load_vectorizer():
    path = os.path.join(ART_DIR, "vectorizer.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path}. Hãy chạy ml_pipeline.py trước.")
    vec = joblib.load(path)
    # Hỗ trợ cả dict{'word','char'} hoặc object custom (nhưng project đang dùng dict)
    if isinstance(vec, dict) and "word" in vec and "char" in vec:
        return vec
    raise TypeError("vectorizer.joblib không đúng định dạng dict{'word','char'}.")

def _hstack_transform(vec, texts):
    Xw = vec["word"].transform(texts)
    Xc = vec["char"].transform(texts)
    return hstack([Xw, Xc]).tocsr()

def current_tfidf_dim(vec) -> int:
    # Lấy số chiều hiện tại của TF-IDF (word + char) bằng cách transform 1 mẫu
    Xw = vec["word"].transform(["x"])
    Xc = vec["char"].transform(["x"])
    return Xw.shape[1] + Xc.shape[1]

def latest_clean_csv() -> str:
    paths = sorted(glob.glob(os.path.join(DATA_CLEAN, "clean_log_*.csv")))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file cleaned trong {DATA_CLEAN}. Hãy chạy clean_only.py.")
    return paths[-1]

# --------- SVD loader with auto-rebuild ----------
def load_or_fit_svd(vec, n_components=512, random_state=42):
    svd_path = os.path.join(RL_DIR, "svd_512.joblib")
    curr_dim = current_tfidf_dim(vec)

    if os.path.exists(svd_path):
        svd = joblib.load(svd_path)
        # sklearn >=1.0 có thuộc tính n_features_in_
        svd_dim = getattr(svd, "n_features_in_", None)
        if svd_dim == curr_dim:
            return svd
        # nếu không có attr, thử transform 1 vector zeros với số cột = curr_dim (khó) -> rebuild luôn cho chắc
        print(f"⚠️  SVD feature dim mismatch (cached={svd_dim}, current={curr_dim}). Rebuild SVD...")
    else:
        print("ℹ️  Không thấy svd_512.joblib -> fit SVD lần đầu...")

    # Fit lại SVD từ cleaned mới nhất
    csv_path = latest_clean_csv()
    df = pd.read_csv(csv_path)
    if "normalized_query" not in df.columns:
        # fallback tìm cột SQL rồi normalize
        for cand in ["sql", "query", "statement", "command"]:
            if cand in df.columns:
                df["normalized_query"] = df[cand].astype(str).map(normalize_sql)
                break
        else:
            raise KeyError("Không có cột normalized_query/sql/query/statement/command trong cleaned CSV.")
    texts = df["normalized_query"].astype(str).tolist()

    # Biến đổi TF-IDF full corpus -> fit SVD
    X = _hstack_transform(vec, texts)
    # đảm bảo n_components <= số cột
    k = min(n_components, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    svd.fit(X)
    joblib.dump(svd, svd_path)
    print(f"✅ Đã fit & lưu SVD ({k} comps) -> {svd_path}")
    return svd

# --------- Public: transform 1 câu SQL cho RL ----------
def transform_query_for_rl(sql: str, vec, svd) -> np.ndarray:
    q = normalize_sql(sql)
    X = _hstack_transform(vec, [q])
    Z = svd.transform(X)  # (1, k)
    return Z.astype(np.float32)
