# Biến thể CA-TCC với DINOv3 cho âm thanh

**Cập nhật phục vụ app 18/10/2026:** Đã bổ sung `scripts/export_dinov3_audio.py` để xuất CPU TorchScript và frontend/normalization/nhãn lớp. Cả RAVDESS và SpeechCommands đã xuất và qua kiểm tra parity/API bằng WAV thật. SpeechCommands hoàn tất ngân sách rút gọn 4 epoch mỗi pha, test accuracy 31.02%, macro-F1 23.54%. ONNX Runtime Web chạy cả hai trên trình duyệt sau khi xuất bản. Hướng dẫn tích hợp tại [web/README.md](../../web/README.md), trạng thái tại `artifacts/dinov3_app_bundles/package_status.json`. Các mô tả “chưa hỗ trợ export” phía dưới là lịch sử trước khi có exporter mới.

**Cập nhật 17/10/2026:** Đã tải checkpoint ViT-S/16 chính thức từ email cấp quyền, xác minh hash, nạp strict và chạy kiểm tra GPU pretrained thành công. Đã khởi chạy tìm tham số với 10% nhãn. Batch 8 chạy được ở cả hai dataset; RAVDESS batch 32 hết VRAM. Xem [quy trình tìm tham số 10% và trạng thái hiện tại](SEARCH.md).

Các phần dưới ghi lại thiết kế và kế hoạch ban đầu; các câu “chưa chạy/chưa tải” phản ánh thời điểm tạo bản thiết kế. Trạng thái thực thi hiện tại được cập nhật trong `SEARCH.md` và các artifact được dẫn tại đó.

Theo yêu cầu, giữ nguyên đường chạy CNN. Không tự khởi chạy biến thể khi baseline kết thúc; các lệnh dưới đây chỉ là hướng dẫn dành cho giai đoạn sau. Cả entry point huấn luyện và pipeline đều chặn chạy khi baseline `full_audio_1p_seed0_20261017` chưa hoàn thành hai bộ, export và WAV parity. Lệnh `--help` chỉ hiển thị trợ giúp, không khởi tạo mô hình.

## 1. Chỗ được bổ sung và thay đổi so với code cũ

Tất cả mã mới nằm trong `variants/dinov3/`. Không sửa `main.py`, `models/model.py`, `models/TC.py`, `trainer/trainer.py`, `dataloader/`, cấu hình dữ liệu hoặc các script chạy baseline trong lần bổ sung này. Chỉ thêm một liên kết tài liệu ở README gốc.

| Tệp mới | Nơi tương ứng ở baseline | Phần thay đổi |
|---|---|---|
| `model.py` | `models/model.py` | Thay ba khối CNN bằng DINOv3 ViT-S/16; thêm chuyển log-mel thành đầu vào 3 kênh, chuyển patch token thành chuỗi thời gian, projection 384 → 128 và classifier |
| `main.py` | `main.py` | Entry point riêng; thêm tham số DINOv3, kiểm soát chính sách đóng băng, xác thực thư mục dữ liệu riêng, lưu hash repo/trọng số và chặn chạy sớm |
| `trainer.py` | `trainer/trainer.py` | Bản sao trainer tại thời điểm tạo biến thể; thay đoạn cố định `conv_block1/2/3.eval()` bằng đóng băng toàn bộ encoder DINOv3 cho linear evaluation |
| `common.py` | Không có | Chặn chạy trước baseline, chuẩn bị dữ liệu riêng, lưu snapshot/hash và ghi JSON nguyên tử |
| `pipeline.py` | `scripts/run_audio_pipeline.py` | Điều phối các pha ở entry point mới, nhãn giả riêng, resume và sinh báo cáo bằng script báo cáo chung |
| `compare.py` | Không có | Đọc hai lượt đã hoàn thành; đối chiếu cùng tỷ lệ nhãn/seed/epoch/manifest; sinh MD, CSV, JSON và biểu đồ sau này |
| `CHANGES.patch` | Hai file main/trainer baseline | Diff để xem trực tiếp những dòng khác giữa baseline và bản sao của biến thể |
| `implementation_manifest.json` | Không có | Hash mã baseline được giữ nguyên và danh sách file bổ sung, trạng thái chưa thực thi |

`models/TC.py`, `models/loss.py`, `dataloader/dataloader.py`, `dataloader/augmentations.py` và cấu hình SpeechCommands/RAVDESS được dùng chung. Các thay đổi tương lai ở những module dùng chung cũng ảnh hưởng biến thể; snapshot ở đầu mỗi experiment giúp truy vết mã thực tế.

## 2. Luồng dữ liệu và vị trí thay encoder

```text
Tensor log-mel chuẩn hóa bằng thống kê train [B,40,T]
  → clip cố định [-4,4], ánh xạ về [0,1]
  → resize số học, lặp thành 3 kênh và chuẩn hóa RGB cố định
  → DINOv3 ViT-S/16: lấy x_norm_patchtokens
  → khôi phục lưới [B, tần số patch, thời gian patch, 384]
  → lấy trung bình theo trục tần số
  → Conv1d 1×1: 384 → 128
  → chuỗi [B,128,L]
      ├── TC + temporal/contextual contrastive loss hiện có
      └── flatten → logits → phân loại
```

Không lấy CLS token để thay chuỗi thời gian. CLS và register tokens không được đưa vào adapter. Code dựa trên patch token theo thứ tự hàng của triển khai ViT chính thức.

| Bộ dữ liệu | Tensor đầu vào | Kích thước ảnh số học mặc định | Lưới patch | Chuỗi đầu ra |
|---|---|---|---|---|
| Speech Commands | `[B,40,101]` | `[B,3,64,240]` | `4×15` | `[B,128,15]` |
| RAVDESS | `[B,40,601]` | `[B,3,64,1232]` | `4×77` | `[B,128,77]` |

Chiều rộng bằng `16 × config.features_len`, chiều cao mặc định 64. Hai chiều đều chia hết cho patch size 16. Đây là thiết kế adapter của dự án, không phải recipe âm thanh chính thức của DINOv3. Không tạo PNG, không dùng colormap và không cần tiền xử lý lại WAV. Resize làm thay đổi độ phân giải thời gian/tần số, cần ghi nhận khi phân tích kết quả.

Chuẩn hóa RGB dùng mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` cố định; không học min/max từ validation hoặc test. Clip z-score và resize nằm trong model, sau augmentation log-mel dùng chung. Hiệu quả của cách chuyển miền này chưa được kiểm chứng.

## 3. Chính sách học và các pha được giữ

Pipeline vẫn chạy:

```text
self_supervised
→ train_linear_{p}p
→ ft_{p}p
→ gen_pseudo_labels
→ SupCon
→ train_linear_SupCon_{p}p
→ ft_SupCon_{p}p
```

Có thể thêm `supervised` bằng `--include-supervised`. Tỷ lệ nhãn hỗ trợ: 1%, 5%, 10%, 50%, 75%. Self-supervised dùng toàn bộ train không nhãn; linear/fine-tune dùng subset nhãn thật; SupCon dùng toàn bộ train với nhãn giả.

| Chính sách | Trong self-supervised / SupCon / fine-tune | Trong linear |
|---|---|---|
| `--backbone-policy trainable` (mặc định) | DINOv3 và projection được cập nhật; classifier học ở pha phân loại | DINOv3 và projection đều đóng băng; chỉ classifier học |
| `--backbone-policy frozen` | DINOv3 luôn đóng băng; projection học qua loss CA-TCC; classifier học ở pha phân loại | DINOv3 và projection đều đóng băng; chỉ classifier học |

Khi linear, encoder luôn ở eval để không thay đổi dropout/trạng thái. Với chính sách frozen, riêng DINOv3 luôn ở eval và chạy không gradient; projection vẫn có gradient ở các pha khác. Vì vậy `frozen` **không được mô tả là fine-tuning toàn bộ DINOv3**. Không bỏ qua self-supervised hoặc SupCon trong pipeline chỉ vì dùng backbone pretrained.

- `--initialization pretrained`: bắt buộc trọng số chính thức ViT-S/16 có sẵn trên đĩa, nạp strict; không tự tải mạng và không lặng lẽ chuyển sang random nếu thiếu file.
- `--initialization random`: khởi tạo kiến trúc DINOv3 không có trọng số pretrained; không truyền `--dinov3-weights`. Đây là đối chứng kiến trúc có thể dùng sau này.
- Adam, learning rate từ config, weight decay `3e-4`, scheduler, augmentation và công thức loss lấy từ baseline. Self-supervised giữ hệ số temporal/contextual `1/0.7`; SupCon giữ `0.01/0.1`, kể cả khác biệt code/bài báo đã ghi nhận trước đó.
- Checkpoint phân loại chọn theo validation cross entropy; self-supervised/SupCon dùng checkpoint cuối. Resume lưu optimizer, scheduler và RNG theo trainer baseline.

## 4. Tách dữ liệu và kết quả

Tên experiment bắt buộc bắt đầu `dinov3_`. Pipeline chỉ liên kết tensor train/val/test và subset nhãn cần thiết từ nguồn; sao chép manifest/normalization nhỏ. Nhãn giả và diagnostics tạo mới ở thư mục riêng:

```text
datasets/tensors/experiments/dinov3_<ten_luot>/<dataset>/
experiments_logs/dinov3_<ten_luot>/<dataset>/<stage>_seed_<seed>/
artifacts/dinov3_<ten_luot>/
```

Không ghi đè nhãn giả, checkpoint, cấu hình hay metrics CNN 1%/10%. Chưa tạo các thư mục kết quả này vì chưa chạy biến thể.

Mỗi lượt tương lai lưu source snapshot, hash mã, phiên bản torch, hash manifest, cấu hình DINOv3 và hash checkpoint pretrained/mã nguồn local DINOv3. Không cho resume cùng lượt nếu các cấu hình DINOv3 hoặc hash nguồn/trọng số đổi.

## 5. Chuẩn bị sau khi lượt hiện tại hoàn thành

**Chưa thực hiện các bước trong phần này.**

1. Đợi baseline 1% hiện tại hoàn thành đầy đủ, gồm export và kiểm tra parity. Không sửa file trạng thái để vượt chốt chặn.
2. Dùng môi trường Python riêng, ví dụ `.venv-dinov3`; không nâng cấp torch trong `.venv` baseline đang chạy.
3. Chuẩn bị bản checkout local của [repo DINOv3 chính thức](https://github.com/facebookresearch/dinov3) và checkpoint **ViT-S/16** chính thức theo điều kiện truy cập của Meta. Không dùng trọng số ViT-B/L hoặc checkpoint teacher dạng khác cho adapter này.
4. Cài dependency của repo DINOv3 tương ứng phiên bản checkout và các thư viện dùng chung của dự án: torch, torchvision, numpy, scipy, scikit-learn, einops, pandas, openpyxl, matplotlib. Yêu cầu upstream tham khảo tại [requirements chính thức](https://github.com/facebookresearch/dinov3/blob/main/requirements.txt). Chưa chốt ma trận phiên bản hay xác nhận tương thích với torch 2.5.1/CUDA của baseline.
5. Chạy kế hoạch kiểm tra ở mục 7 trước khi huấn luyện đầy đủ.

Không có tác vụ nền hay automation nào được thêm để tự chạy DINOv3. Automation báo kết quả hiện có chỉ theo dõi baseline 1%.

## 6. Lệnh dành cho giai đoạn sau, chưa chạy

Chạy từ thư mục gốc `CA-TCC-main`. Thay `/PATH/TO/...` bằng đường dẫn thực. Giữ riêng repo upstream và trọng số, không đưa vào `models/` của baseline.

```bash
# Đối chứng cùng mức 1% nhãn, seed/epoch/batch với CNN.
.venv-dinov3/bin/python variants/dinov3/pipeline.py \
  --experiment dinov3_trainable_1p_seed0 \
  --datasets RAVDESS SpeechCommands \
  --source-data-root datasets/tensors/full \
  --label-percent 1 --epochs 40 --batch-size 32 --seed 0 \
  --dinov3-repo /PATH/TO/dinov3 \
  --dinov3-weights /PATH/TO/dinov3_vits16_official.pth \
  --initialization pretrained --backbone-policy trainable
```

Batch 32 là cấu hình để đối chiếu, **chưa xác nhận vừa VRAM GTX 1650 4 GB**. Trước khi chạy dài cần đo bộ nhớ trên batch nhỏ. Nếu buộc giảm batch, dùng experiment khác, ghi rõ thay đổi vì số negatives trong contrastive loss thay đổi; không tự coi hai cấu hình là đối chứng chỉ khác encoder.

Các biến thể tùy chọn, mỗi biến thể cần tên experiment khác:

- `--backbone-policy frozen`: ít bộ nhớ cho gradient backbone hơn; vẫn cần đo bộ nhớ/tốc độ thực tế.
- `--initialization random`: bỏ tham số weights, để phân biệt phần lợi ích của pretrained với kiến trúc.
- `--label-percent 10`: dùng tên experiment riêng, sau này so với baseline CNN 10% cùng seed.
- `--resume`: thêm vào đúng lệnh trước đó, giữ nguyên cấu hình và dữ liệu.

Pipeline sinh metrics, CSV dự đoán, confusion matrix, đường học, checkpoint best/last/resume qua trainer và báo cáo dùng chung. **Chưa hỗ trợ TorchScript export cho DINOv3**; không dùng `scripts/export_audio_model.py` baseline để nạp checkpoint biến thể. Pipeline mới không tự gọi exporter CNN và không tuyên bố có bundle đã kiểm chứng.

Sau khi cả CNN và DINOv3 đã hoàn thành ở **cùng mức nhãn**:

```bash
.venv-dinov3/bin/python variants/dinov3/compare.py \
  --baseline-experiment full_audio_1p_seed0_20261017 \
  --variant-experiment dinov3_trainable_1p_seed0
```

Lệnh so sánh chỉ đọc kết quả đã có và sinh `comparison/SO_SANH_BACKBONE.md`, `metrics.csv`, `metrics.json`, `backbone_comparison.png` trong artifacts của biến thể. Không tự huấn luyện. Mismatch manifest sẽ dừng để xem xét; nếu đường dẫn được di chuyển giữa các lượt, cần đối chiếu split thực tế và ghi nhật ký trước khi sửa cơ chế này, không bỏ chốt kiểm tra âm thầm.

## 7. Kế hoạch kiểm tra để sau — chưa thực thi

- Xác nhận baseline hiện tại hoàn thành; xác nhận chốt chặn không cho chạy trước thời điểm đó.
- Kiểm tra cú pháp/import với môi trường riêng, dependency và nguồn DINOv3 đã chốt phiên bản.
- Forward cho cả `[B,40,101]` và `[B,40,601]`; xác nhận logits 35/8 lớp, chuỗi 128×15/77, tensor hữu hạn, backward qua TC/loss.
- Kiểm tra chỉ classifier đổi trong linear; projection/backbone đổi đúng policy ở fine-tune và contrastive.
- Kiểm tra thiếu/sai weights phải báo lỗi, không tải mạng hoặc fallback random.
- Kiểm tra dữ liệu/split và subset 1% đúng baseline; pseudo labels chỉ ghi trong thư mục dinov3_ riêng.
- Kiểm tra checkpoint dependency, chọn best theo validation và resume qua nhiều epoch có/không gián đoạn.
- Đo VRAM/thời gian cho hai bộ; đặc biệt RAVDESS có 308 patch tokens. Không suy ra khả năng chạy từ số tham số đơn thuần.
- Sau đó chạy thử nhỏ trong experiment riêng, rồi mới chạy đầy đủ và dùng script compare. Không dùng test để chọn cấu hình.

## 8. Những điều phải ghi trong báo cáo

DINOv3 pretrained là backbone thị giác đã học từ dữ liệu ngoài. Kết quả so với CNN học từ đầu chịu ảnh hưởng của kiến trúc, pretraining và adapter; chưa phải phép đo riêng lợi ích kiến trúc. Cần phân biệt `pretrained/trainable`, `pretrained/frozen` và `random/trainable` nếu thực hiện các đối chứng này.

ViT có self-attention toàn cục: token theo thời gian có thể đã chứa thông tin từ các thời điểm sau trong cùng clip. Adapter giữ hình dạng chuỗi, nhưng **không đảm bảo cơ chế dự đoán tương lai mang cùng ý nghĩa như CNN gốc**. Không gọi biến thể là tương đương hoàn toàn với thiết kế temporal CA-TCC ban đầu.

Giữ split theo người nói, cùng nhãn/seed/ngân sách, báo Accuracy, macro-F1, balanced accuracy, số tham số và tài nguyên. Nếu chỉ có một seed, không kết luận chắc chắn DINOv3 tốt hơn từ một con số.

## 9. Nguồn API dùng để viết adapter

- [Model card DINOv3](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md): ViT-S/16, patch tokens, đầu vào ảnh.
- [Hub backbones chính thức](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py): entry `dinov3_vits16`, `pretrained=False`, cấu trúc pretrained checkpoint.
- [Vision Transformer chính thức](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py): `forward_features`, `x_norm_patchtokens`.

Các API được đọc từ upstream khi triển khai; upstream có thể đổi. Chưa clone hoặc xác minh bằng thực thi với một revision cụ thể. Pipeline/main ghi hash nguồn local thực tế khi chạy sau này.
