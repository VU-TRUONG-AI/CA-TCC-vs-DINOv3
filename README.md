# CA-TCC trên Speech Commands và RAVDESS

Toàn bộ dự án đã gom trong **một thư mục `CA-TCC-main`**: dữ liệu, tiền xử lý, mã mô hình, môi trường chạy, kết quả và tài liệu. Trong VSCode, chỉ cần mở thư mục này.

## Mở đúng nơi theo công việc

| Bạn muốn làm gì? | Mở ở đâu? |
|---|---|
| Xem sơ đồ thư mục và nhật ký dọn dẹp | [docs/SAP_XEP_THU_MUC.md](docs/SAP_XEP_THU_MUC.md) |
| Xem WAV, dữ liệu đã xử lý và mã tiền xử lý | [datasets/README.md](datasets/README.md) |
| Lấy báo cáo và hình tiền xử lý | [datasets/reports/BAO_CAO_TIEN_XU_LY.md](datasets/reports/BAO_CAO_TIEN_XU_LY.md), `datasets/figures/` |
| Lấy kết quả huấn luyện đầy đủ, bảng và hình cho Word | [báo cáo huấn luyện](artifacts/full_audio_10p_seed0_20261017/KET_QUA_HUAN_LUYEN.md) |
| Xem checkpoint và log từng giai đoạn | `experiments_logs/full_audio_10p_seed0_20261017/` |
| Xem thay đổi encoder DINOv3, tham số từng lượt và kết quả 10% nhãn | [README thực nghiệm DINOv3](artifacts/dinov3_search_10p_20261017/README.md) — tự cập nhật từ lượt chạy |
| Lượt DINOv3 SpeechCommands đang chạy để nộp ngày 18/10 | [Tham số RAVDESS chuyển sang SpeechCommands](artifacts/dinov3_speechcommands_ravdess_best_10p_20261018/README.md) — 4 epoch mỗi pha, báo cáo tự cập nhật |
| Xem các thay đổi để chạy với dữ liệu mới | [nhật ký chuyển đổi](docs/NHAT_KY_CHUYEN_DOI_DU_LIEU.md) |
| Chạy lại huấn luyện, đánh giá và suy luận | [hướng dẫn tái lập](docs/REPRODUCE_AUDIO_VI.md) |
| Xem bài báo và slide của bạn | [documents/README.md](documents/README.md) |

## Cấu trúc chính

```text
CA-TCC-main/
├── README.md                 Hướng dẫn đầu tiên
├── datasets/                 Tất cả dữ liệu và quy trình tiền xử lý
│   ├── raw/                  WAV và gói dữ liệu gốc
│   ├── processed/            Đặc trưng NPZ: full và smoke
│   ├── tensors/              Tensor PT và nhãn giả: full và smoke
│   ├── pipeline/             Mã tiền xử lý, kiểm tra, dựng hình
│   ├── results/              Kết quả kiểm tra tiền xử lý
│   ├── reports/              Báo cáo tiền xử lý
│   └── figures/              Hình tiền xử lý đưa vào báo cáo
├── documents/                Bài báo, slide và tài liệu của bạn
├── docs/                     Hướng dẫn, nhật ký thay đổi, tài liệu tham chiếu
├── artifacts/                Báo cáo, CSV, biểu đồ và mô hình đã xuất
├── experiments_logs/         Checkpoint, lịch sử epoch, kết quả từng pha
├── config_files/             Cấu hình mô hình và hai bộ dữ liệu
├── dataloader/               Nạp dữ liệu và augmentation
├── models/                   Encoder, temporal transformer, loss
├── trainer/                  Huấn luyện, đánh giá, sinh nhãn giả
├── scripts/                  Điều phối chạy, export, suy luận, báo cáo
├── tests/                    Kiểm tra mã nguồn
├── web/                      Bản nháp web đang tạm dừng
├── main.py                   Chạy một giai đoạn huấn luyện
└── .venv/                    Môi trường Python dùng chung
```

Dữ liệu chỉ nằm trong **`datasets/`**. Đã bỏ các liên kết `Data`, `data`, `data_smoke`, `runtime_data` ở gốc để tránh nhầm thành nhiều bộ dữ liệu. Các lệnh hiện tại dùng trực tiếp đường dẫn mới.

`config_files/HAR_Configs.py` vẫn cần thiết vì hai cấu hình âm thanh kế thừa tham số chung từ lớp này. Frontend âm thanh dùng chung nằm ở `datasets/pipeline/audio_features.py`.

## Trạng thái hiện tại

Tiền xử lý và kiểm chứng toàn bộ 107.269 WAV đã đạt. Đã hoàn thành huấn luyện trên dữ liệu đầy đủ của cả Speech Commands và RAVDESS bằng GPU GTX 1650, 40 epoch/giai đoạn, 10% nhãn thật cho các bước phân loại ít nhãn, seed 0. Có supervised 100% nhãn làm mốc tham khảo. Đã lưu checkpoint, metrics, export TorchScript và kiểm chứng suy luận WAV.

CA-TCC fine-tune đạt Accuracy **85,48% / macro-F1 84,38% trên Speech Commands**, **23,75% / macro-F1 20,05% trên RAVDESS**. Đây là một seed và một split, chưa đo độ biến thiên qua nhiều lượt.

- [Kết quả huấn luyện đầy đủ](artifacts/full_audio_10p_seed0_20261017/KET_QUA_HUAN_LUYEN.md).
- [Đánh giá Speech Commands](artifacts/full_audio_10p_seed0_20261017/DANH_GIA_SPEECHCOMMANDS.md).

- [Nhật ký thay đổi để repo chạy với dữ liệu mới](docs/NHAT_KY_CHUYEN_DOI_DU_LIEU.md).
- [Kết quả chạy thử và giới hạn diễn giải](artifacts/smoke_audio_20261017/KET_QUA_CHAY_THU.md).
- [Trạng thái và bằng chứng chạy](artifacts/RUN_STATUS.md).

```bash
# Từ thư mục repo; experiment mới cho mỗi lần kiểm chứng mới.
.venv/bin/python scripts/verify_audio_repo.py --experiment smoke_audio_new --device cuda:0
```

Dùng `--device cpu` nếu không có GPU; dùng `--resume` với cùng experiment/cấu hình khi tiếp tục lượt đã chạy. Phần web hiện tạm dừng để ưu tiên repo huấn luyện.

Mã cũ đã nén tại `docs/reference/legacy_20261013.tar.gz`. Nhật ký sắp xếp và kiểm chứng nằm trong [SẮP XẾP THƯ MỤC](docs/SAP_XEP_THU_MUC.md).

Đây là áp dụng CA-TCC sang âm thanh; không đồng nhất kết quả với số liệu benchmark HAR/EEG trong bài báo gốc.

## Lượt bổ sung 1% nhãn thật đã hoàn tất

Hoàn tất cả hai bộ lúc 15:33 ngày 17/10/2026: đủ 40 epoch/pha, seed 0, toàn bộ test. CA-TCC fine-tune đạt Accuracy/macro-F1 **53,00%/48,07% trên Speech Commands**, **14,58%/8,53% trên RAVDESS**. Đã kiểm tra checkpoint, metrics và parity 43 WAV. [Bảng so sánh đầy đủ 1%–10%](artifacts/full_audio_1p_seed0_20261017/SO_SANH_1P_10P.md), [CSV](artifacts/full_audio_1p_seed0_20261017/SO_SANH_1P_10P.csv), [biểu đồ](artifacts/full_audio_1p_seed0_20261017/SO_SANH_1P_10P.png). Lượt 1% có gián đoạn và resume sau epoch 39 self-supervised Speech Commands; xem nhật ký khôi phục trong artifacts.

## Biến thể DINOv3 — chỉ bổ sung mã, chưa chạy

Theo yêu cầu của giảng viên, mã thay encoder bằng DINOv3 được tách riêng tại `variants/dinov3/`, giữ nguyên đường chạy CNN. [README DINOv3 riêng](variants/dinov3/README.md) chỉ rõ file mới, adapter, các chế độ học và lệnh dành cho giai đoạn sau. Chưa tải trọng số, cài môi trường, chạy kiểm tra hay huấn luyện biến thể; mã có chốt chặn đến khi baseline 1% hiện tại hoàn thành.
