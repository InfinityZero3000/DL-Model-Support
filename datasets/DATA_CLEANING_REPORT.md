# Data Cleaning Report — LexiLingo Training Dataset

**Date:** 2026-04-29  
**Script:** `preprocess_data.py`  
**Input:** `training_data/` → **Output:** `training_data_clean/`

---

## 1. Pipeline tiền xử lý

Toàn bộ quá trình được thực hiện bằng `preprocess_data.py`, chạy theo thứ tự cố định trên từng sample. File gốc **không bị chỉnh sửa** — output ghi vào folder `training_data_clean/` riêng.

### Bước 1 — Parse & validate JSON

Mỗi dòng JSONL được parse và kiểm tra cấu trúc tối thiểu:
- Phải có `task`, `messages` (list), ít nhất 1 message role `"user"` và 1 role `"assistant"`
- `content` của assistant phải là chuỗi JSON hợp lệ (parse được bằng `json.loads`)
- Dòng lỗi parse bị bỏ qua và ghi vào counter `json_parse_error`

### Bước 2 — Deduplication (near-duplicate removal)

**Khóa dedup:** `(task, user_content[:200])` — lấy 200 ký tự đầu của user message để khớp các sample gần giống nhau (cùng đoạn văn, khác phần cuối không đáng kể).

**Chiến lược shared set:** Một `seen_keys` set dùng chung cho cả 2 split, xử lý train trước rồi val sau. Nhờ đó:
- Trong train: loại bỏ các sample trùng lặp nội bộ.
- Trong val: loại bỏ cả các sample đã xuất hiện ở train → **ngăn data leakage từ val vào train**.

Kết quả: 795 mẫu train và 102 mẫu val bị loại bỏ (~3.7% và ~7.2%).

### Bước 3 — Fix theo task

> **Lưu ý quan trọng:** Script **không xóa từ, không sửa câu văn**. Nó chỉ sửa **cấu trúc JSON** của phần trả lời (`assistant content`). Nội dung ngôn ngữ (câu chữ, bản dịch, giải thích) được giữ nguyên 100%.

#### Task `vocabulary`

Mỗi sample vocabulary có dạng:
```
User:    "Analyze the vocabulary in: 'She was utterly devastated...'"
Assistant: '{"level": "C1", "key_words": ["devastated", "utterly"], "explanation": "..."}'
```

Script parse phần `assistant content` từ chuỗi JSON → dict Python, sửa dict, rồi serialize lại thành chuỗi JSON. **Câu văn của user và explanation không bị chạm vào.**

| Lỗi | Trước (lỗi) | Sau (đúng) | Nguyên nhân |
|-----|-------------|-----------|-------------|
| `key_words` **thiếu** (4,050 train / 267 val) | `{"level":"C1","explanation":"..."}` | `{"level":"C1","key_words":[],"explanation":"..."}` | Data generator bỏ sót field |
| `key_words` là **string** (1,635 train / 106 val) | `{"level":"C1","key_words":"devastated, utterly","explanation":"..."}` | `{"level":"C1","key_words":[],"explanation":"..."}` | Metadata từ crawler bị bleed thành string thay vì list |

> Tại sao để `[]` thay vì giữ string split? Vì không thể tái tạo lại list gốc chính xác từ chuỗi (không rõ delimiter, viết hoa/thường, từ ghép). Dùng `[]` an toàn hơn là inject dữ liệu sai.

#### Task `fluency`

Mỗi sample fluency có dạng:
```
User:    "Rate the fluency of: 'He go to school yesterday.'"
Assistant: '{"fluency_score": 2, "feedback": "Subject-verb agreement error..."}'
```

Lỗi: 487 train / 34 val sample có thêm trường `reasoning` do một phiên bản prompt cũ sinh ra:
```json
// Trước (lỗi):
{"fluency_score": 2, "feedback": "...", "reasoning": "The sentence contains..."}

// Sau (đúng):
{"fluency_score": 2, "feedback": "..."}
```

Script chỉ xóa key `reasoning` khỏi dict — `fluency_score` và `feedback` không bị chạm vào.

#### Task `grammar`, `dialogue`

Không có lỗi schema phát hiện được — sample đi thẳng qua sau dedup, không sửa gì.

### Bước 4 — Ghi output

Mỗi sample sạch được serialize lại bằng `json.dumps(..., ensure_ascii=False)` và ghi theo định dạng JSONL (mỗi dòng 1 sample). File `clean_report.json` lưu toàn bộ stats per-split để audit.

---

## 2. Vấn đề phát hiện & số liệu

| # | Task | Vấn đề | Train | Val |
|---|------|---------|------:|----:|
| 1 | vocabulary | `key_words` bị **thiếu** hoàn toàn | 4,050 mẫu | 267 mẫu |
| 2 | vocabulary | `key_words` là **string** thay vì list (metadata bleed) | 1,635 mẫu | 106 mẫu |
| 3 | fluency | Có trường `reasoning` thừa → schema không nhất quán | 487 mẫu | 34 mẫu |
| 4 | all | Near-duplicate theo `(task, user_content[:200])` | 795 mẫu | 102 mẫu |

---

## 3. Kết quả sau clean

| Split | Trước | Sau | Giảm |
|-------|------:|----:|-----:|
| train | 21,504 | **20,709** | -795 (-3.7%) |
| val | 1,412 | **1,310** | -102 (-7.2%) |
| **Tổng** | **22,916** | **22,019** | **-897** |

### Phân bố task (sau clean)

| Task | Train | Val |
|------|------:|----:|
| dialogue | 4,507 | 247 |
| fluency | 5,810 | 381 |
| grammar | 4,707 | 309 |
| vocabulary | 5,685 | 373 |

---

## 4. Kiểm tra schema cuối cùng (final audit)

| Kiểm tra | Train | Val |
|----------|------:|----:|
| vocabulary thiếu `key_words` | **0** ✅ | **0** ✅ |
| vocabulary `key_words` sai kiểu | **0** ✅ | **0** ✅ |
| fluency có `reasoning` | **0** ✅ | **0** ✅ |
| Non-JSON assistant response | **0** ✅ | **0** ✅ |

---

## 5. Phân bố độ dài token (ước tính `chars / 4`)

| Ngưỡng | Train | Val |
|--------|------:|----:|
| > 512 tokens | 524 (2.5%) | 34 (2.6%) |
| > 1,024 tokens | 103 (0.5%) | 6 (0.5%) |
| > **2,048 tokens** | **41 (0.20%)** | **4 (0.31%)** |

> `MAX_SEQ_LENGTH=2048` trong notebook sẽ truncate 45 mẫu (0.20% tổng) — không đáng kể, không cần lọc thêm.

---

## 6. Files output

```
training_data_clean/
├── train_clean.jsonl   # 20,709 mẫu, ~16 MB
├── val_clean.jsonl     # 1,310 mẫu,  ~1.0 MB
└── clean_report.json   # Stats chi tiết
```

---

## 7. Việc cần làm tiếp theo

- [ ] Upload `training_data_clean/` lên Kaggle Datasets (`lexilingo-datasets-clean`)
- [ ] Cập nhật notebook cell 15: đổi `train.jsonl` → `train_clean.jsonl`, `val.jsonl` → `val_clean.jsonl`
