# ml_pipeline.py (FIXED)
# Train supervised ML: TF-IDF (word+char) -> LinearSVC (binary) + LogisticRegression (reward)
# Tự phát hiện file clean_* mới nhất, tự xử lý thiếu cột/NaN.

import re
import os
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

DATA_DIR = "data/cleaned"
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def pick_latest_clean_file():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "clean_log_*.csv")))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file cleaned trong {DATA_DIR}. Hãy chạy clean_only.py trước.")
    return paths[-1]

def normalize_minimal(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"'[^']*'", " 'str' ", s)
    s = re.sub(r'"[^"]*"', ' "str" ', s)
    s = re.sub(r'\b\d+\b', ' 0 ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalized_query
    if "normalized_query" not in df.columns:
        for cand in ["sql", "query", "statement", "command"]:
            if cand in df.columns:
                df["normalized_query"] = df[cand].astype(str).map(normalize_minimal)
                break
        else:
            raise KeyError("Thiếu cột normalized_query/sql/query/statement/command.")
    df["normalized_query"] = df["normalized_query"].astype(str)

    # risky
    if "risky" not in df.columns or df["risky"].isna().any():
        if "reward" in df.columns:
            df["risky"] = (pd.to_numeric(df["reward"], errors="coerce") < 0).astype("Int64")
        else:
            patt = r"\b(drop|truncate)\b|(\bdelete\b(?!.*\bwhere\b))|(\bupdate\b(?!.*\bwhere\b))|(\bgrant\b|\brevoke\b|\bcreate\s+user\b)"
            df["risky"] = df["normalized_query"].str.contains(patt, regex=True, flags=re.IGNORECASE).astype("Int64")

    before = len(df)
    df = df.dropna(subset=["normalized_query", "risky"]).copy()
    df["risky"] = df["risky"].astype("Int64")
    df = df[df["risky"].isin([0, 1])]
    removed = before - len(df)
    if removed > 0:
        print(f"⚠️  Loại {removed} dòng thiếu nhãn/thiếu query.")
    return df

def build_vectorizers(texts):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.995, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 3),
        min_df=2, max_df=0.995, sublinear_tf=True,
    )
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    X = hstack([Xw, Xc]).tocsr()
    vec = {"word": word_vec, "char": char_vec}
    return X, vec

def transform_with_vectorizer(vec, texts):
    Xw = vec["word"].transform(texts)
    Xc = vec["char"].transform(texts)
    return hstack([Xw, Xc]).tocsr()

def main():
    warnings.filterwarnings("ignore")
    csv_path = pick_latest_clean_file()
    print(f"📄 Load: {csv_path}")
    df = pd.read_csv(csv_path)
    df = ensure_columns(df)

    texts = df["normalized_query"].astype(str).values
    y_bin = df["risky"].astype(int).values

    # reward (multiclass) nếu có
    y_mcls = None
    if "reward" in df.columns:
        y_mcls_series = pd.to_numeric(df["reward"], errors="coerce")
        valid_mask = ~y_mcls_series.isna()
        if valid_mask.sum() > 1:
            df = df.loc[valid_mask].copy()
            texts = df["normalized_query"].astype(str).values
            y_bin = df["risky"].astype(int).values
            y_mcls = df["reward"].astype(int).values
        else:
            y_mcls = None  # không đủ lớp

    # split cho binary
    X_train_txt, X_test_txt, yb_train, yb_test = train_test_split(
        texts, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    # fit TF-IDF trên train
    X_train, vec = build_vectorizers(X_train_txt)
    X_test = transform_with_vectorizer(vec, X_test_txt)

    # Binary: LinearSVC
    clf_bin = LinearSVC()
    clf_bin.fit(X_train, yb_train)
    yb_pred = clf_bin.predict(X_test)
    acc_bin = accuracy_score(yb_test, yb_pred)

    report_lines = []
    report_lines.append(f"Binary accuracy: {acc_bin:.4f}\n")
    report_lines.append("Binary classification report:\n")
    report_lines.append(classification_report(yb_test, yb_pred, digits=4))
    report_lines.append("Confusion matrix (binary):\n")
    report_lines.append(str(confusion_matrix(yb_test, yb_pred)))

    # Multiclass: LogisticRegression (nếu có)
    clf_m = None
    if y_mcls is not None and len(np.unique(y_mcls)) > 1:
        # Đồng bộ split cho reward: tạo mặt nạ dựa trên chuỗi train
        mcls_texts = texts  # numpy array
        # mặt nạ: những câu thuộc X_train_txt (so sánh bằng nội dung chuỗi)
        mask_train_bool = pd.Index(mcls_texts).isin(X_train_txt)  # -> numpy bool array
        # tách tập
        X_train_m = transform_with_vectorizer(vec, mcls_texts[mask_train_bool])
        X_test_m  = transform_with_vectorizer(vec, mcls_texts[~mask_train_bool])
        y_m = y_mcls  # numpy int array
        y_train_m = y_m[mask_train_bool]
        y_test_m  = y_m[~mask_train_bool]

        clf_m = LogisticRegression(max_iter=200, multi_class="auto")
        clf_m.fit(X_train_m, y_train_m)
        ym_pred = clf_m.predict(X_test_m)
        acc_m = accuracy_score(y_test_m, ym_pred)

        report_lines.append("\n\nMulticlass (reward) accuracy: {:.4f}\n".format(acc_m))
        report_lines.append("Multiclass classification report:\n")
        report_lines.append(classification_report(y_test_m, ym_pred, digits=4))
        joblib.dump(clf_m, os.path.join(ART_DIR, "model_multiclass.joblib"))
    else:
        report_lines.append("\n\nMulticlass (reward): SKIPPED (không có hoặc reward chỉ một lớp)\n")

    # Lưu artifacts
    joblib.dump(vec, os.path.join(ART_DIR, "vectorizer.joblib"))
    joblib.dump(clf_bin, os.path.join(ART_DIR, "model_binary.joblib"))
    with open(os.path.join(ART_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("✅ Saved artifacts ->", ART_DIR)
    print("   - vectorizer.joblib (dict{'word','char'})")
    print("   - model_binary.joblib (LinearSVC)")
    if clf_m is not None:
        print("   - model_multiclass.joblib (LogisticRegression)")
    print("   - report.txt")

if __name__ == "__main__":
    main()
