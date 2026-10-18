# DINOv3 trên SpeechCommands — dùng trực tiếp tham số thắng RAVDESS

Experiment: `dinov3_speechcommands_ravdess_best_10p_20261018`. Trạng thái: **completed**. Cập nhật: 2026-10-18T15:05:13+07:00.

## 1. Mục tiêu và thay đổi theo yêu cầu

Người dùng yêu cầu lấy trực tiếp bộ hyperparameter tốt nhất của RAVDESS để huấn luyện SpeechCommands. Đã dừng grid SpeechCommands cũ và giữ lại config, history, checkpoint, nhãn giả, log của lượt dở dang. Không tiếp tục tìm 5 tổ hợp còn lại hoặc chạy refinement. Lượt này khởi tạo lại từ DINOv3 pretrained chính thức; chỉ chuyển hyperparameter, không chuyển weights đã học trên RAVDESS hay nối weights từ lượt SpeechCommands LR=1e-5.

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

### 3.1. Các lần thử thành công trên RAVDESS

Các lượt dưới đây đều dùng DINOv3 ViT-S/16 pretrained, 10% nhãn, batch 8, seed 0 và 10 epoch mỗi pha. Kết quả là validation tại checkpoint được chọn của pha cuối `ft_SupCon_10p`.

| Lượt | Learning rate | Weight decay | Epoch chọn | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|---:|
| t00 | 1e-05 | 0.0001 | 6 | 1.806568 | 33.75% |
| t01 | 1e-05 | 0.0003 | 3 | 1.850943 | 27.92% |
| t02 | 3e-05 | 0.0001 | 1 | 1.846157 | 27.08% |
| t03 | 3e-05 | 0.0003 | 2 | 1.822803 | 24.58% |
| t04 | 0.0001 | 0.0001 | 9 | 1.765119 | 28.33% |
| t05 | 0.0001 | 0.0003 | 8 | 1.702606 | 31.25% |

Nguồn số liệu: [trials.csv](../dinov3_search_10p_20261017/trials.csv).

### 3.2. Kết luận lựa chọn và áp dụng cho SpeechCommands

**Trong 6 tổ hợp đã thử, t05 là cấu hình tốt nhất theo tiêu chí validation loss thấp nhất: LR=1e-4, weight decay=3e-4**, loss **1.702606**, validation accuracy **31.25%**, checkpoint epoch **8**. Lượt t00 có accuracy validation cao nhất (**33.75%**) nhưng loss cao hơn, nên không được chọn theo tiêu chí đã định. Không dùng test để chọn tham số.

Bộ tham số t05 được dùng trực tiếp để huấn luyện **SpeechCommands**, giữ batch **8**, seed **0**, nhãn **10%**; giảm ngân sách xuống **4 epoch mỗi pha** để kịp hạn nộp. SpeechCommands khởi tạo lại từ DINOv3 pretrained chính thức, không nạp weights đã fine-tune trên RAVDESS.

| Dataset áp dụng | LR | Weight decay | Epoch/pha | Validation loss pha cuối | Validation accuracy | Test accuracy | Test macro-F1 | Test balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SpeechCommands | 1e-4 | 3e-4 | 4 | 2.245166 | 32.24% | 31.02% | 23.54% | 25.25% |

Đánh giá test trên **11.005 mẫu**, với checkpoint `ft_SupCon_10p` được chọn ở epoch **4** theo validation loss. Đây là cấu hình được chọn từ các thử nghiệm RAVDESS để sử dụng cho SpeechCommands; chưa chứng minh tối ưu trên SpeechCommands. Kết quả từng pha ở mục 4 và test cuối ở mục 5.

### 3.3. Tham số khởi tạo của lượt SpeechCommands

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

Giữ nguyên split/subset SpeechCommands đã chuẩn bị. Lượt hoàn tất dùng LR=1e-4, weight decay=3e-4 và ngân sách 4 epoch mỗi pha, khởi tạo độc lập từ pretrained chính thức. Ngân sách rút gọn theo yêu cầu nộp lúc 19:00 ngày 18/10/2026, để dành thời gian tích hợp web/API; không coi là lượt đối chứng cùng epoch với RAVDESS. LR có thể được scheduler giảm; không suy đoán lịch LR từng epoch.

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

Chưa ghi nhận lỗi ở lượt mới.

Cấu hình được chọn trên dataset khác; không bảo đảm tốt nhất trên SpeechCommands. Một seed và pretraining thị giác ngoài dataset không đủ để kết luận thống kê. Batch 8 khác baseline CNN batch 32. Đã xuất TorchScript CPU và ONNX cho cả hai mô hình, có kiểm tra parity.

## 8. Gói xuất và nhúng app

Cả hai mô hình DINOv3 đã xuất TorchScript, kiểm tra logits/WAV parity và API localhost bằng WAV thật. [Gói model](../dinov3_app_models.zip), [app demo kèm model](../dinov3_demo_app.zip), [hướng dẫn API](../../web/README.md). Suy luận CPU ở backend Python.

## 9. So sánh và bàn giao

| Dataset | CNN gốc: accuracy / macro-F1 | DINOv3: accuracy / macro-F1 | Epoch mỗi pha CNN / DINOv3 |
|---|---|---|---|
| RAVDESS | 23.75% / 20.05% | 24.17% / 18.54% | 40 / 40 |
| SpeechCommands | 85.48% / 84.38% | 31.02% / 23.54% | 40 / 4 |

Cùng pha cuối ft_SupCon_10p, seed 0 và tập test được giữ cố định. Batch CNN 32, DINOv3 8. SpeechCommands dùng ngân sách rút gọn và tham số chuyển từ RAVDESS; chưa chứng minh hiệu quả hơn baseline. Không dùng kết quả test này để tiếp tục chỉnh tham số. Không coi softmax là xác suất đã hiệu chỉnh.

Web chạy ONNX Runtime Web trên thiết bị người dùng, hoạt động khi máy huấn luyện tắt: https://ca-tcc-audio-studio-vu-20261018.royalcream8.chatgpt.site. App ZIP chạy TorchScript với API Python /api/predict. Trạng thái xuất bản thực tế: ../audio_studio_web_20261018/deployment_state.json. Giao diện tuần tự, responsive, có chuyển đổi file thu âm sang WAV và tải CSV/WAV miễn phí; hỗ trợ codec tùy trình duyệt.

Bàn giao 18/10/2026: cả hai mô hình đã deploy thành công (Site version 4), SpeechCommands browser/API parity và CSV đã qua; hai ZIP đã kiểm tra toàn vẹn. Giữ giao diện tuần tự và chức năng chuyển WAV miễn phí.
