# LexiLingo - Hướng dẫn Training

Có 2 notebook tương ứng với 2 nền tảng:

| File | Nền tảng |
|------|----------|
| `finetune_qwen_lora.v3.0.ipynb` | Google Colab |
| `finetune_qwen_lora_kaggle.v1.0.ipynb` | Kaggle |

Cả hai đều cùng model, cùng thông số training, chỉ khác phần lưu trữ và secrets.

---

## Thông số chung

| Tham số | Giá trị |
|---------|---------|
| Model | Qwen/Qwen3-1.7B |
| LoRA rank / alpha | 16 / 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| MAX_SEQ_LENGTH | 512 |
| Epochs | 5 |
| Batch size (per device) | 3 |
| Gradient accumulation | 6 (effective batch = 18) |
| Learning rate | 3e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Optimizer | paged_adamw_8bit |
| Save / eval steps | 150 |
| Early stopping patience | 5 (dừng sau 750 steps không cải thiện) |
| Precision | fp16 (T4/V100) hoặc bf16 (A100) |

---

## Google Colab - `finetune_qwen_lora.v3.0.ipynb`

### Yêu cầu

- Tài khoản Google (Drive để lưu checkpoint)
- Colab Free hoặc Colab Pro
- GPU: T4 (free) hoặc A100 (Pro+)

### Bước 1 - Upload dữ liệu lên Google Drive

```
MyDrive/
  LexiLingo/
    training_data/
      downloaded_datasets/
        train.jsonl
        val.jsonl
        split_report.json      (tùy chọn)
```

Cách nhanh nhất: kéo thả folder `datasets/datasets/` từ máy vào Drive.

### Bước 2 - Mở notebook trên Colab

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. File > Upload notebook > chọn `finetune_qwen_lora.v3.0.ipynb`

### Bước 3 - Cài GPU

Runtime > Change runtime type > Hardware accelerator > **T4 GPU** (hoặc A100)

### Bước 4 - Chạy

Run All (Ctrl+F9) hoặc chạy từng cell từ trên xuống.

Quy trình:
1. Cell 2: cài packages + unsloth (3-5 phút)
2. Cell 3: mount Google Drive - cấp quyền khi được hỏi
3. Cell 11: kiểm tra output dir đã tới Drive chưa
4. Cell 13: load model Qwen3-1.7B từ HuggingFace (~3 phút)
5. Cell 15: load và format dataset
6. Cell 17: bắt đầu training + auto-resume nếu có checkpoint

### Bước 5 - Resume sau khi disconnect

Khi Colab bị mất kết nối hoặc hết quota, chỉ cần:

1. Mở lại notebook
2. Mount Drive lại (cell 3)
3. Chạy từ cell 4 trở đi
4. Cell 17 sẽ tự động phát hiện checkpoint mới nhất và tiếp tục

### Checkpoint

Lưu vào Drive tại:

```
MyDrive/LexiLingo/unified_model_optimized/
  checkpoint-150/
  checkpoint-300/
  training_state.json
  unified_lora_adapter/    (sau khi train xong)
```

### Ước tính thời gian

| GPU | ~27k samples, 5 epochs |
|-----|------------------------|
| T4 (free) | 12 - 14 giờ |
| A100 (Pro+) | 3 - 4 giờ |

---

## Kaggle - `finetune_qwen_lora_kaggle.v1.0.ipynb`

### Yêu cầu

- Tài khoản Kaggle (free)
- GPU: T4 x2 hoặc P100 (cả hai đều miễn phí, 30 giờ/tuần)
- Internet bật trong session settings

### Bước 1 - Upload dữ liệu thành Kaggle Dataset

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) > New Dataset
2. Đặt tên: `lexilingo-datasets`
3. Upload các file:
   - `train.jsonl`
   - `val.jsonl`
   - `split_report.json` (tùy chọn)
4. Publish dataset (có thể đặt private)

Dữ liệu sẽ được mount tại `/kaggle/input/lexilingo-datasets/`.

### Bước 2 - Tạo notebook mới

1. Vào [kaggle.com/code](https://www.kaggle.com/code) > New Notebook
2. File > Import Notebook > chọn `finetune_qwen_lora_kaggle.v1.0.ipynb`

### Bước 3 - Gắn dataset vào notebook

Trong notebook editor: **+ Add Data** > tìm `lexilingo-datasets` > Add.

### Bước 4 - Cài đặt Session

Settings (biểu tượng bánh răng góc phải):

| Tùy chọn | Giá trị |
|----------|---------|
| Accelerator | GPU T4 x2 hoặc P100 |
| Internet | ON (bắt buộc) |

### Bước 5 - Cài Secrets (tùy chọn)

Để bật WandB hoặc MongoDB, vào Account > Settings > Secrets > Add New Secret.

| Secret name | Giá trị |
|-------------|---------|
| `WANDB_API_KEY` | Key từ wandb.ai |
| `MONGODB_URI` | MongoDB Atlas URI |

Notebook tự động phát hiện và bật logging nếu có secret.

### Bước 6 - Chạy

Run All.

Quy trình:
1. Cell 2: cài packages + unsloth
2. Cell 3: phát hiện Kaggle paths + thông tin GPU
3. Cell 11: config OUTPUT_DIR = `/kaggle/working/model/outputs/unified`
4. Cell 13: load Qwen3-1.7B
5. Cell 15: load `/kaggle/input/lexilingo-datasets/train.jsonl`
6. Cell 17: training

### Bước 7 - Lưu kết quả

File trong `/kaggle/working/` tồn tại đến hết session nhưng sẽ mất khi session kết thúc.

Các cách lưu trước khi session kết thúc:

**Cách 1 - Save version (khuyến nghị):**

Session > Save Version > Save & Run All > sau đó tải output từ version history.

**Cách 2 - Download trực tiếp:**

Output panel (cột phải) > tìm `model/outputs/unified/` > Download.

**Cách 3 - Upload lên HuggingFace Hub:**

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/kaggle/working/model/outputs/unified/unified_lora_adapter",
    repo_id="your-username/lexilingo-lora",
    repo_type="model",
)
```

### Checkpoint

Checkpoint lưu tại:

```
/kaggle/working/
  model/
    outputs/
      unified/
        checkpoint-150/
        checkpoint-300/
        training_state.json
        unified_lora_adapter/
```

Lưu ý: checkpoint chỉ resume được trong cùng session. Nếu session kết thúc, phải download checkpoint trước khi hết giờ.

### Ước tính thời gian

| GPU | ~27k samples, 5 epochs |
|-----|------------------------|
| T4 x2 | 5 - 6 giờ |
| P100 | 7 - 8 giờ |
| T4 x1 | 10 - 12 giờ |

---

## So sánh 2 nền tảng

| Tiêu chí | Google Colab | Kaggle |
|----------|-------------|--------|
| GPU miễn phí | T4 (12h/session) | T4 x2 hoặc P100 (30h/tuần) |
| Lưu trữ persistent | Google Drive (15 GB free) | Phải download trước khi hết session |
| Resume sau disconnect | Tự động qua Drive | Chỉ resume trong cùng session |
| Tốc độ | T4: ~12-14h | T4 x2: ~5-6h |
| Internet | Luôn có | Phải bật thủ công |
| Secrets | Không có built-in | Có Kaggle Secrets |
| Data upload | Upload lên Drive | Tạo Kaggle Dataset |

---

## Early Stopping

Cả hai notebook đều có EarlyStoppingCallback với `patience=5`.

Training tự động dừng nếu `eval_loss` không cải thiện sau 5 lần đánh giá liên tiếp:

```
5 x eval_steps = 5 x 150 = 750 steps (~0.5 epoch)
```

Sau khi dừng, model tốt nhất (theo eval_loss) được load lại tự động do `load_best_model_at_end=True`.

---

## Sau khi train xong

Adapter được lưu tại:

- Colab: `MyDrive/LexiLingo/unified_model_optimized/unified_lora_adapter/`
- Kaggle: `/kaggle/working/model/outputs/unified/unified_lora_adapter/`

Kích thước adapter: ~80 MB (không phải full model).

Để chạy inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", ...)
model = PeftModel.from_pretrained(model, "./unified_lora_adapter")
```
