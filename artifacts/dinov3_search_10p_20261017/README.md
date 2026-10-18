# CA-TCC với DINOv3 ViT-S/16 — thực nghiệm 10% nhãn

> **Chuyển hướng theo yêu cầu ngày 18/10:** Đã dừng tìm tham số SpeechCommands, giữ các kết quả cũ bên dưới. Lượt mới đã hoàn tất, dùng trực tiếp LR=1e-4, weight decay=3e-4 thắng trên RAVDESS, batch 8, seed 0, ngân sách 4 epoch mỗi pha để dành thời gian web/API trước hạn 19:00. Xem [README và kết quả lượt mới](../dinov3_speechcommands_ravdess_best_10p_20261018/README.md).

Experiment: `dinov3_search_10p_20261017`. Trạng thái ghi nhận: **interrupted**. README cập nhật: 2026-10-18T10:40:45+07:00.

README được sinh từ cấu hình và kết quả đã lưu; dấu **—** nghĩa là chưa có số liệu, không phải 0. Kết quả từng lượt đang chạy còn tạm thời. Chỉ kết luận cấu hình thắng khi có đủ kết quả lựa chọn.

## 1. Mục tiêu và phạm vi

Thay encoder CNN của CA-TCC bằng backbone thị giác DINOv3 ViT-S/16, chuyển log-mel thành đầu vào ảnh trong model và giữ đầu ra chuỗi cho temporal/contextual contrastive learning. Thực nghiệm trên RAVDESS và SpeechCommands với 10% nhãn thật; chọn hyperparameter bằng validation.

## 2. Thay encoder/backbone ở đâu và như thế nào?

| File | Thay đổi và vai trò |
|---|---|
| [variants/dinov3/model.py](../../variants/dinov3/model.py) | `DinoSequenceEncoder` thay 3 conv block bằng factory chính thức `dinov3.hub.backbones.dinov3_vits16`; `DinoAudioModel` ghép encoder, projection và classifier. |
| [variants/dinov3/main.py](../../variants/dinov3/main.py) | Khởi tạo model biến thể; nạp checkpoint strict; truyền policy, kích thước adapter, learning rate/weight decay; cấu hình đóng băng theo pha; lưu resolved config/hash. |
| [variants/dinov3/trainer.py](../../variants/dinov3/trainer.py) | Trainer riêng dựa trên baseline; linear probe đóng băng toàn encoder, chọn checkpoint bằng validation; hỗ trợ tắt đánh giá test trong search. |
| [variants/dinov3/pipeline.py](../../variants/dinov3/pipeline.py) | Điều phối các pha và dữ liệu/nhãn giả riêng từng experiment. |
| [variants/dinov3/search.py](../../variants/dinov3/search.py) | Thử grid tuần tự, lưu CSV/JSON, chọn winner và huấn luyện lại; đánh giá test cuối. |
| [scripts/report_dinov3_search.py](../../scripts/report_dinov3_search.py) | Sinh/cập nhật README này từ dữ liệu thực tế đã ghi. |

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

## 3. Khởi tạo ban đầu

| Thành phần | Giá trị |
|---|---|
| Backbone / initialization | DINOv3 ViT-S/16 / `pretrained` |
| Pretraining | LVD-1689M; kiến thức học từ ảnh ngoài dataset âm thanh |
| Checkpoint chính thức | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| File weights | `/home/truongvu/Documents/CĐ/CA-TCC-main/weights/dinov3_vits16.pth` |
| SHA-256 | `08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d` |
| Cách nạp | State dict chính thức, `strict=True`; thiếu/sai weights báo lỗi |
| Projection và classifier | Khởi tạo mới theo mặc định PyTorch, dưới seed của lượt chạy |
| Temporal model TC | Khởi tạo mới; không lấy weights CNN cũ |
| Backbone policy | `trainable`; khi linear probe, toàn encoder gồm projection đóng băng |
| Seed / label budget | 0 / 10% train có nhãn |
| Batch / trial epochs / final epochs | 8 / 10 / 40 mỗi pha học |
| Optimizer | Adam; betas=(0.9, 0.99) |
| Scheduler | ReduceLROnPlateau theo validation loss ở pha phân loại; không step ở self-supervised/SupCon |
| Adapter / loss temperature | image_height=64; clip_z=4; temperature=0.2 |

Cấu hình gốc dataset kế thừa lr=3e-4, weight decay baseline=3e-4, batch=32. Đợt DINOv3 này override LR/weight decay bằng từng tổ hợp dưới đây và batch=8 theo phép đo VRAM. Lượt đầu khởi tạo optimizer với lr=1e-5, weight decay=1e-4. LR trong bảng là LR khởi tạo optimizer cho cả model và TC; scheduler có thể giảm LR về sau. Không suy đoán lịch LR từng epoch khi history không ghi trường đó; trạng thái optimizer nằm trong `resume.pt`.

Self-supervised giữ hệ số temporal/contextual 1/0.7; SupCon 0.01/0.1. Các tensor trọng số học được lưu trong checkpoint, không liệt kê hàng triệu giá trị trong README. Resolved config đầy đủ nằm ở `run_config.json` của từng pha.

## 4. Các tham số đã thay đổi và kết quả từng lượt

Grid: learning rate = 1e-05, 3e-05, 0.0001; weight decay = 0.0001, 0.0003. Mỗi tổ hợp chạy lại đầy đủ pipeline; không nối weights từ trial trước. 10% nhãn dùng ở linear/fine-tune; self-supervised dùng toàn train không nhãn, SupCon dùng toàn train với pseudo labels riêng.

| Dataset | Trial | LR ban đầu | Weight decay | Batch | Epoch/pha | Seed | Trạng thái | Val loss chọn | Val accuracy | Epoch chọn |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| RAVDESS | t00 | 1e-05 | 0.0001 | 8 | 10 | 0 | completed | 1.80657 | 33.75% | 6 |
| RAVDESS | t01 | 1e-05 | 0.0003 | 8 | 10 | 0 | completed | 1.85094 | 27.92% | 3 |
| RAVDESS | t02 | 3e-05 | 0.0001 | 8 | 10 | 0 | completed | 1.84616 | 27.08% | 1 |
| RAVDESS | t03 | 3e-05 | 0.0003 | 8 | 10 | 0 | completed | 1.8228 | 24.58% | 2 |
| RAVDESS | t04 | 0.0001 | 0.0001 | 8 | 10 | 0 | completed | 1.76512 | 28.33% | 9 |
| RAVDESS | t05 | 0.0001 | 0.0003 | 8 | 10 | 0 | completed | 1.70261 | 31.25% | 8 |
| SpeechCommands | t00 | 1e-05 | 0.0001 | 8 | 10 | 0 | interrupted | — | — | — |
| SpeechCommands | t01 | 1e-05 | 0.0003 | 8 | 10 | 0 | chưa chạy | — | — | — |
| SpeechCommands | t02 | 3e-05 | 0.0001 | 8 | 10 | 0 | chưa chạy | — | — | — |
| SpeechCommands | t03 | 3e-05 | 0.0003 | 8 | 10 | 0 | chưa chạy | — | — | — |
| SpeechCommands | t04 | 0.0001 | 0.0001 | 8 | 10 | 0 | chưa chạy | — | — | — |
| SpeechCommands | t05 | 0.0001 | 0.0003 | 8 | 10 | 0 | chưa chạy | — | — | — |

Các cột validation ở bảng trên lấy từ checkpoint được chọn của pha `ft_SupCon_10p`. Trial chưa kết thúc sẽ chưa có điểm xếp hạng. Chi tiết: [trials.csv](trials.csv), [search_status.json](search_status.json).

## 5. Tiến độ và kết quả từng pha đã ghi

| Experiment / dataset | Pha | Epoch tham chiếu | Train loss | Val loss | Val accuracy | Loại mốc |
|---|---|---:|---:|---:|---:|---|
| dinov3_search_10p_20261017_ravdess_final / RAVDESS | SupCon | 40 | 0.0492934 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_final / RAVDESS | ft_10p | 10 | 1.45458 | 1.71103 | 25.00% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_final / RAVDESS | ft_SupCon_10p | 3 | 1.62282 | 1.76819 | 22.50% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_final / RAVDESS | self_supervised | 40 | 1.36959 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_final / RAVDESS | train_linear_10p | 14 | 1.87885 | 1.76889 | 28.33% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_final / RAVDESS | train_linear_SupCon_10p | 18 | 1.51271 | 1.75003 | 29.58% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t00 / RAVDESS | SupCon | 10 | 0.0542844 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t00 / RAVDESS | ft_10p | 8 | 1.50377 | 1.83558 | 30.83% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t00 / RAVDESS | ft_SupCon_10p | 6 | 1.18412 | 1.80657 | 33.75% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t00 / RAVDESS | self_supervised | 10 | 1.35029 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t00 / RAVDESS | train_linear_10p | 10 | 1.82478 | 1.93097 | 26.67% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t00 / RAVDESS | train_linear_SupCon_10p | 8 | 1.5076 | 1.85073 | 27.92% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t01 / RAVDESS | SupCon | 10 | 0.0536882 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t01 / RAVDESS | ft_10p | 8 | 1.46328 | 1.86143 | 30.42% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t01 / RAVDESS | ft_SupCon_10p | 3 | 1.51534 | 1.85094 | 27.92% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t01 / RAVDESS | self_supervised | 10 | 1.46225 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t01 / RAVDESS | train_linear_10p | 10 | 1.83934 | 1.92709 | 22.92% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t01 / RAVDESS | train_linear_SupCon_10p | 8 | 1.47369 | 1.84767 | 30.83% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t02 / RAVDESS | SupCon | 10 | 0.0401966 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t02 / RAVDESS | ft_10p | 5 | 1.32392 | 1.82825 | 26.25% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t02 / RAVDESS | ft_SupCon_10p | 1 | 1.71321 | 1.84616 | 27.08% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t02 / RAVDESS | self_supervised | 10 | 0.855819 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t02 / RAVDESS | train_linear_10p | 4 | 1.8257 | 1.97352 | 20.83% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t02 / RAVDESS | train_linear_SupCon_10p | 2 | 1.59925 | 1.90162 | 22.50% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t03 / RAVDESS | SupCon | 10 | 0.0386952 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t03 / RAVDESS | ft_10p | 4 | 1.5849 | 1.73757 | 28.33% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t03 / RAVDESS | ft_SupCon_10p | 2 | 1.48811 | 1.8228 | 24.58% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t03 / RAVDESS | self_supervised | 10 | 0.599503 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t03 / RAVDESS | train_linear_10p | 10 | 1.63073 | 1.74576 | 30.83% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t03 / RAVDESS | train_linear_SupCon_10p | 4 | 1.52837 | 1.82791 | 27.08% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t04 / RAVDESS | SupCon | 10 | 0.0573372 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t04 / RAVDESS | ft_10p | 7 | 1.66692 | 1.76545 | 22.92% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t04 / RAVDESS | ft_SupCon_10p | 9 | 1.48387 | 1.76512 | 28.33% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t04 / RAVDESS | self_supervised | 10 | 1.87719 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t04 / RAVDESS | train_linear_10p | 10 | 1.79247 | 1.8136 | 30.42% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t04 / RAVDESS | train_linear_SupCon_10p | 6 | 1.74286 | 1.7876 | 28.33% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t05 / RAVDESS | SupCon | 10 | 0.0596898 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t05 / RAVDESS | ft_10p | 2 | 1.91357 | 1.76239 | 22.50% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t05 / RAVDESS | ft_SupCon_10p | 8 | 1.62453 | 1.70261 | 31.25% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t05 / RAVDESS | self_supervised | 10 | 1.66496 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_ravdess_t05 / RAVDESS | train_linear_10p | 10 | 1.81633 | 1.81284 | 26.25% | checkpoint được chọn |
| dinov3_search_10p_20261017_ravdess_t05 / RAVDESS | train_linear_SupCon_10p | 8 | 1.74851 | 1.76124 | 26.25% | checkpoint được chọn |
| dinov3_search_10p_20261017_speechcommands_t00 / SpeechCommands | SupCon | 2 | 0.0140521 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_speechcommands_t00 / SpeechCommands | ft_10p | 9 | 0.325972 | 0.532681 | 85.38% | checkpoint được chọn |
| dinov3_search_10p_20261017_speechcommands_t00 / SpeechCommands | self_supervised | 10 | 0.181872 | — | — | epoch cuối đã ghi |
| dinov3_search_10p_20261017_speechcommands_t00 / SpeechCommands | train_linear_10p | 10 | 3.44023 | 3.44372 | 6.35% | checkpoint được chọn |

Self-supervised/SupCon không có validation accuracy phân loại; giá trị placeholder 0 trong console không được coi là kết quả đánh giá. Lịch sử đầy đủ từng epoch trong `history.json`.

## 6. Cấu hình thắng và đánh giá test cuối

Tiêu chí: validation cross entropy nhỏ nhất tại checkpoint được chọn của pha cuối; không dùng test để chọn tham số. Winner được huấn luyện lại từ pretrained đủ 40 epoch mỗi pha, rồi chọn checkpoint theo validation và đánh giá test.

| Dataset | Trial được chọn | LR / weight decay | Val loss lượt cuối | Test accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|
| RAVDESS | dinov3_search_10p_20261017_ravdess_t05 | 0.0001 / 0.0003 | 1.76819 | 24.17% | 18.54% | 22.66% |
| SpeechCommands | chưa chọn | — | — | — | — | — |

“Tốt nhất” chỉ trong grid, seed và ngân sách đã thử; chưa phải tối ưu toàn cục. Xếp hạng 10 epoch có thể khác 40 epoch. Một seed chưa đủ kết luận thống kê.

## 7. Các lượt lỗi và lần thử lại

Chưa ghi nhận trial lỗi.

## 8. Lưu trữ và tái lập

- `experiments_logs/<trial>/<dataset>/<stage>_seed_0/`: config, history, selection, best/last/resume checkpoint.
- `artifacts/<trial>/`: source snapshot/hash, package versions và pipeline log.
- `datasets/tensors/experiments/<trial>/`: liên kết dữ liệu, manifest/normalization và pseudo labels riêng.
- `best_parameters_<dataset>.json`: cấu hình thắng; `winner_test_metrics.json`/`winner_test_predictions.csv` nằm trong pha cuối của finalist.
- Trọng số model và hash nguồn/checkpoint trong config/snapshot cho phép truy vết từng lượt.

Lệnh resume dưới đây chỉ là tài liệu tái lập grid cũ; không tiếp tục grid đã dừng theo yêu cầu khi bàn giao này:

```bash
.venv-dinov3/bin/python -u variants/dinov3/search.py --resume
```

Sinh lại README từ kết quả đã lưu:

```bash
.venv-dinov3/bin/python scripts/report_dinov3_search.py --experiment dinov3_search_10p_20261017
```

## 9. Điều kiện diễn giải

DINOv3 có pretraining ngoài dataset; adapter resize/clipping và attention toàn cục cũng thay đổi. Batch 8 khác CNN baseline batch 32, ảnh hưởng số negatives và số bước optimizer. Không quy toàn bộ chênh lệch cho encoder hoặc coi hai lượt chỉ khác backbone. Đã bổ sung exporter TorchScript và ONNX riêng, kiểm tra thành công cả hai dataset; xem báo cáo lượt mới và web/README.md.

Nguồn kiến trúc: [DINOv3 chính thức](https://github.com/facebookresearch/dinov3). Thiết kế chi tiết: [README biến thể](../../variants/dinov3/README.md).
