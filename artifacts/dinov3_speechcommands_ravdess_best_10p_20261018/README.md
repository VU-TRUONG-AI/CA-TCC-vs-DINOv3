# DINOv3 trên SpeechCommands — dùng trực tiếp tham số thắng RAVDESS

## 1. Mục tiêu

Chỉ dùng DINOv3 làm một phương hướng nghiên cứu không lấy DINOv3 làm backbone mặc định cho CA-TCC trên audio:

Lệch miền: DINOv3 được pretrain trên ảnh tự nhiên RGB. Spectrogram không có cấu trúc như ảnh (cạnh, vật thể, texture), và hai trục thời gian/tần số mang ý nghĩa vật lý khác nhau.
Augmentation không khớp: Multi-crop, flip, color jitter của DINO làm hỏng nội dung spectrogram, nên lợi thế pretrain bị giảm nhiều.
Không khớp thiết kế CA-TCC: Temporal Contrasting và Contextual Contrasting cần chuỗi latent theo thời gian, còn ViT cho token patch 2D, phải gộp theo trục tần số mới dùng được.

Vì thời gian huấn luyện lâu nê lấy trực tiếp bộ hyperparameter tốt nhất của RAVDESS để huấn luyện SpeechCommands.

## 2. Thay encoder/backbone ở đâu và như thế nào?
| File | Thay đổi và vai trò |
|---|---|
| [variants/dinov3/model.py](../../variants/dinov3/model.py) | `DinoSequenceEncoder` thay 3 conv block bằng factory chính thức `dinov3.hub.backbones.dinov3_vits16`; `DinoAudioModel` ghép encoder, projection và classifier. |
| [variants/dinov3/main.py](../../variants/dinov3/main.py) | Khởi tạo model biến thể; nạp checkpoint strict; truyền policy, kích thước adapter, learning rate/weight decay; cấu hình đóng băng theo pha; lưu resolved config/hash. |
| [variants/dinov3/trainer.py](../../variants/dinov3/trainer.py) | Trainer riêng dựa trên baseline; linear probe đóng băng toàn encoder, chọn checkpoint bằng validation; hỗ trợ tắt đánh giá test trong search. |
| [variants/dinov3/pipeline.py](../../variants/dinov3/pipeline.py) | Điều phối các pha và dữ liệu/nhãn giả riêng từng experiment. |
| [variants/dinov3/search.py](../../variants/dinov3/search.py) | Thử grid tuần tự, lưu CSV/JSON, chọn winner và huấn luyện lại; đánh giá test cuối. |
| [scripts/report_dinov3_fixed.py](../../scripts/report_dinov3_fixed.py) | Sinh báo cáo lượt cố định này; báo cáo grid trước do scripts/report_dinov3_search.py tạo. |

Đường chạy CNN gốc ở `main.py`, `models/model.py` và `trainer/trainer.py` vẫn tồn tại. Biến thể dùng chung `models/TC.py`, `models/loss.py`, dataloader, augmentation và cấu hình dataset.

```text
Log-mel đã chuẩn hóa train [B,40,T]
 → clip [-4,4], ánh xạ [0,1]
 → resize, lặp 3 kênh, chuẩn hóa RGB cố định
 → DINOv3 ViT-S/16 → x_norm_patchtokens (không dùng CLS/register tokens)
 → khôi phục lưới tần số × thời gian, trung bình theo tần số
 → projection Conv1d 1×1: 384 → 128 → chuỗi [B,128,L]
    ├─ temporal/contextual contrastive losses của CA-TCC
    └─ flatten → Linear classifier → logits
```

| Dataset | Log-mel | Đầu vào DINOv3 | Chuỗi sau projection | Số lớp |
|---|---|---|---|---:|
| RAVDESS | `[B,40,601]` | `[B,3,64,1232]` | `[B,128,77]` | 8 |
| SpeechCommands | `[B,40,101]` | `[B,3,64,240]` | `[B,128,15]` | 35 |

Đây là adapter âm thanh của dự án; không phải recipe âm thanh chính thức của DINOv3. RGB mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]. Không tạo PNG hoặc colormap.

## 3. Nguồn lựa chọn và tham số khởi tạo

Nguồn: `dinov3_search_10p_20261017_ravdess_t05`. Validation loss của trial nguồn: **1.702606**. Đây là cấu hình thắng trên RAVDESS theo validation cross entropy, chưa được xác nhận tối ưu cho SpeechCommands.

| Tham số | Giá trị |
|---|---|
| Learning rate ban đầu | 0.0001 |
| Weight decay | 0.0003 |
| Batch / seed | 8 / 0 |
| Epoch mỗi pha học | 4 |
| Initialization | DINOv3 ViT-S/16 pretrained LVD-1689M, strict loading |
| Chính sách | Trainable; linear probe đóng băng encoder và projection |
| Optimizer | Adam, betas=(0.9,0.99); LR/weight decay trên áp dụng cho model và TC |
| Scheduler | ReduceLROnPlateau theo validation ở pha phân loại |
| Nhãn | 10% train có nhãn cho linear/fine-tune; toàn train cho contrastive |
| Adapter | image height=64, clip z=4, RGB normalization cố định |
| Weights đầu vào | `weights/dinov3_vits16.pth`; projection/classifier/TC khởi tạo mới dưới seed 0 |

Cùng split/subset SpeechCommands đã dùng trước đó. Lượt cũ có LR=1e-5, weight decay=1e-4, 10 epoch mỗi pha; lượt mới đổi LR=1e-4, weight decay=3e-4 và ngân sách 4 epoch mỗi pha.
## 4. Kết quả từng pha

| Pha | Trạng thái | Epoch tham chiếu | Train loss | Val loss | Val accuracy | Mốc |
|---|---|---:|---:|---:|---:|---|
| self_supervised | completed | 4 | 0.420467 | — | — | epoch cuối đã ghi |
| train_linear_10p | completed | 3 | 3.56212 | 3.54261 | 4.59% | checkpoint chọn |
| ft_10p | completed | 4 | 1.87323 | 1.58226 | 53.17% | checkpoint chọn |
| gen_pseudo_labels | completed | — | — | — | — | epoch cuối đã ghi |
| SupCon | completed | 4 | 0.0248713 | — | — | epoch cuối đã ghi |
| train_linear_SupCon_10p | completed | 4 | 2.84738 | 2.79742 | 20.31% | checkpoint chọn |
| ft_SupCon_10p | completed | 4 | 2.24084 | 2.24517 | 32.24% | checkpoint chọn |

Dấu — nghĩa là chưa có hoặc không áp dụng. Self-supervised/SupCon không có accuracy phân loại. Validation chọn checkpoint; test không được dùng để điều chỉnh cấu hình.

## 5. Đánh giá test cuối

| Accuracy | Macro-F1 | Balanced accuracy | Số mẫu |
|---:|---:|---:|---:|
| 31.02% | 23.54% | 25.25% | 11005 |

Chỉ đánh giá test sau khi hoàn tất pipeline, trên checkpoint pha `ft_SupCon_10p` được chọn theo validation loss. Kết quả chưa có không được coi là 0.

## 6. Tái lập và artifact

```bash
.venv-dinov3/bin/python -u scripts/run_dinov3_fixed.py --resume
```

- `fixed_run_status.json`: cấu hình cố định, nguồn lựa chọn và lịch sử attempt.
- `supervisor.log`, `pipeline.log`: log tiến trình.
- `experiments_logs/<experiment>/SpeechCommands/<stage>_seed_0/`: resolved config, history, selection, best/last/resume checkpoints.
- Pha cuối chứa `winner_test_metrics.json`, `winner_test_predictions.csv` và hash checkpoint được đánh giá.
- `source_snapshot.zip`, `source_sha256.json`, `packages.log`: provenance do pipeline lưu.
- [Kết quả các lượt trước](../dinov3_search_10p_20261017/README.md) được giữ nguyên, có ghi rõ dừng theo yêu cầu.

## 7. Lỗi và giới hạn

Do thời gian gấp rút và qua trình huấn luyện tiêu tốn thời gian nên cấu hình được chọn trên RAVDESS dataset; không bảo đảm tốt nhất trên SpeechCommands. Một seed và pretraining thị giác ngoài dataset không đủ để kết luận thống kê. Batch 8 khác baseline CNN batch 32. 

## 8. So sánh và bàn giao

| Dataset | CNN gốc: accuracy / macro-F1 | DINOv3: accuracy / macro-F1 | Epoch mỗi pha CNN / DINOv3 |
|---|---|---|---|
| RAVDESS | 23.75% / 20.05% | 24.17% / 18.54% | 40 / 40 |
| SpeechCommands | 85.48% / 84.38% | 31.02% / 23.54% | 40 / 4 |

Cùng pha cuối ft_SupCon_10p, seed 0 và tập test được giữ cố định. Batch CNN 32, DINOv3 8. SpeechCommands dùng ngân sách rút gọn và tham số chuyển từ RAVDESS; Từ kết quả huấn luyện có thể thấy dùng DINOv3 thay cho encoder làm giảm độ chính xác của mô hình chính vì vậy sẽ chỉ dừng lại ở mức nghiên cứu không áp dụng vào sản phẩm cuối.