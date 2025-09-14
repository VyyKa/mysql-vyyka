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