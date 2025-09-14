# Database Log Analysis Project

Quy trình gồm **3 bước chính** với 3 script:  

---

## 1. Parse log MySQL → CSV
Chạy script `parse_log.py` để đọc file log gốc (`.log`) và tạo file CSV trong thư mục **`pare_result/`**.

```bat
python parse_log.py
````

👉 Kết quả: `pare_result/parsed log N.csv` (N tự tăng).

---

## 2. Clean CSV đã parse

Chạy script `clean_only.py` để làm sạch file parsed (thêm `normalized_query`, `query_type`, ép kiểu reward).
File output được lưu trong **`clean file/`**.

```bat
python clean_only.py --input "pare_result\parsed log 1.csv"
```

👉 Kết quả: `clean file/clean log N.csv`.

---

## 3. Train AI model

Chạy script `ml_pipeline.py` với file clean để vector hóa, huấn luyện model, và sinh báo cáo.
Kết quả lưu trong **`mlpineline_result/`**.

```bat
python ml_pipeline.py --input "clean file\clean log 1.csv"
```

👉 Kết quả:

* `clean_log_final.csv` (file clean đã dùng để train)
* `vectorizer.joblib` (TF-IDF vectorizer)
* `model_binary.joblib` (model phân loại risky/benign)
* `model_multiclass.joblib` (model dự đoán reward)
* `report.txt` (báo cáo metrics)

---

## 📂 Cấu trúc thư mục

```
prj/
│
├─ parse_log.py
├─ clean_only.py
├─ ml_pipeline.py
├─ requirements.txt
│
├─ pare_result/          # kết quả từ parse_log.py
├─ clean file/           # kết quả từ clean_only.py
├─ mlpineline_result/    # kết quả từ ml_pipeline.py
└─ report/               # (tùy chọn, báo cáo cũ)
```

---

## ⚙️ Cài đặt môi trường

Cài thư viện cần thiết:

```bat
pip install -r requirements.txt
```

---

## 🚀 Quy trình nhanh

1. `parse_log.py` → `pare_result\parsed log N.csv`
2. `clean_only.py` → `clean file\clean log N.csv`
3. `ml_pipeline.py` → `mlpineline_result\*` (model + báo cáo)

```

---

Bạn có muốn mình tạo sẵn file **`README.md`** này để bạn tải về luôn, hay để bạn copy-paste vào Notepad tự lưu?
```
