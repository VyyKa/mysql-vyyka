# Database Log Analysis Project

---

## 🚀 Checklist Demo Project MySQL Log AI

### 1. Bật các dịch vụ nền tảng

- Mở **Docker Desktop** → chờ trạng thái Running.
- Vào project:
  ```bat
  cd C:\Users\Kha\Downloads\prj\DBS_project
  docker compose up -d
  docker ps
  ```
- Đảm bảo có 3 container: **elasticsearch**, **logstash**, **kibana**.
- Kiểm tra nhanh Kibana:
  - Trình duyệt: [http://localhost:5601](http://localhost:5601)
  - Data View: `mysql-logs-rl*` (time field = `@timestamp`) đã tồn tại.

---

### 2. Khởi động MySQL (Windows service)

```bat
net start MySQL80
```

- Kiểm tra log đang ghi:
  ```sql
  SHOW VARIABLES LIKE 'general_log';
  SHOW VARIABLES LIKE 'general_log_file';
  ```
- Log file phải tăng trong `DBS_project/data/raw_logs/prj1.log`.

---

### 3. Pipeline Python (nếu muốn demo training lại)

*(Không bắt buộc trong demo, chỉ nếu muốn show training)*

- Parse và clean log:
  ```bat
  python parse_log.py
  python clean_only.py
  ```
- Train supervised ML:
  ```bat
  python ml_pipeline.py
  ```
- Train RL (sẽ lâu hơn, có thể bỏ qua trong demo):
  ```bat
  python -m rl.train_rl
  ```
- Artifacts sẽ sinh vào `artifacts/`.

---

### 4. Demo Inference Realtime

- Chạy inference bằng CLI:
  ```bat
  python -m rl.infer_rl "DELETE FROM users;"
  python -m rl.infer_rl "UPDATE t SET a=1;"
  python -m rl.infer_rl "UPDATE t SET a=1 WHERE id=1;"
  ```
- Kết quả console hiển thị:
  ```
  SQL: ...
  Predicted: risky (rule override)
  Q-values: [...]
  ```
- Đồng thời ghi vào file `logs/rl_pred.jsonl`.
- Kiểm tra file log:
  ```powershell
  Get-Content .\logs\rl_pred.jsonl -Tail 5
  ```

---

## Quy trình gốc

### 1. Parse log MySQL → CSV

Chạy script `parse_log.py` để đọc file log gốc (`.log`) và tạo file CSV trong thư mục **`pare_result/`**.

```bat
python parse_log.py
```

👉 Kết quả: `pare_result/parsed log N.csv` (N tự tăng).

---

### 2. Clean CSV đã parse

Chạy script `clean_only.py` để làm sạch file parsed (thêm `normalized_query`, `query_type`, ép kiểu reward).
File output được lưu trong **`clean file/`**.

```bat
python clean_only.py --input "pare_result\parsed log 1.csv"
```

👉 Kết quả: `clean file/clean log N.csv`.

---

### 3. Train AI model

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
├─ artifacts/            # kết quả RL training/inference
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

---

Bạn có muốn mình tạo sẵn file **`README.md`** này để bạn tải về luôn, hay để bạn copy-paste vào Notepad tự lưu?
