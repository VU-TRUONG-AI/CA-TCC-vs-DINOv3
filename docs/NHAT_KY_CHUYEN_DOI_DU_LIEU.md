# Nhật ký đưa CA-TCC chạy với Speech Commands và RAVDESS

Ngày thực hiện đợt kiểm chứng: 17/10/2026. Mục tiêu của đợt này là xác nhận repo chạy được trên hai bộ âm thanh mới, từ dữ liệu đã xử lý đến huấn luyện, đánh giá, xuất mô hình và suy luận WAV. Chạy thử một epoch không phải benchmark chất lượng.

## 1 Phần đã có khi bắt đầu đợt này

- Toàn bộ 105.829 WAV Speech Commands và 1.440 WAV RAVDESS đã được tiền xử lý và kiểm tra ở workspace `datasets/` (được gộp về dự án ngày 17/10/2026).
- `Data`, `data`, `data_smoke` là liên kết đến dữ liệu gốc, đặc trưng đầy đủ và tập kiểm tra nhỏ. Chưa có PyTorch hoặc checkpoint được xác nhận chạy thành công.
- Hai cấu hình `SpeechCommands_Configs.py` và `RAVDESS_Configs.py` đã có: đầu vào 40 kênh log-mel; lần lượt 35/8 lớp; chiều đặc trưng cuối encoder 15/77. Cấu hình âm thanh kế thừa các tham số chung từ `HAR_Configs.py`.
- Frontend WAV đã dùng mono, 16 kHz, log-mel 40 kênh, FFT 512, cửa sổ 400 mẫu, hop 160 mẫu; thời lượng 1 giây/6 giây. Chuẩn hóa bằng thống kê train, không dùng val/test để ước lượng mean/std.
- DataLoader, augmentation âm thanh, pipeline nhiều giai đoạn, export và predict đã được viết nhưng trước đây mới kiểm tra cú pháp. Không ghi nhận những phần này là kết quả huấn luyện.

### Các điều chỉnh để chuyển từ HAR sang âm thanh

| Thành phần | Cấu hình HAR được giữ làm base | Speech Commands | RAVDESS |
|---|---|---|---|
| Kênh đầu vào CNN | 9 kênh cảm biến | 40 kênh log-mel | 40 kênh log-mel |
| Số lớp | 6 | 35 từ lệnh | 8 cảm xúc |
| Chiều thời gian đầu vào | Theo dữ liệu HAR | 101 frame | 601 frame |
| Chiều đặc trưng sau encoder | 18 | 15 | 77 |
| Dạng tensor | `[N,C,T]` | `[N,40,101]` | `[N,40,601]` |
| Augmentation | Scaling, jitter, permutation | Nhiễu nhỏ và time/frequency masking | Nhiễu nhỏ và time/frequency masking |

CNN 1D vẫn chạy trên trục thời gian; mỗi kênh log-mel đóng vai trò một chuỗi đầu vào. Giữ kiến trúc CA-TCC và các loss, điều chỉnh cấu hình cho đầu vào âm thanh. Không dùng phép đảo đoạn lời nói của cấu hình cảm biến làm augmentation mặc định.

Các sửa lỗi tích hợp đã có từ đợt trước: nạp dữ liệu trên CPU và chuyển đúng thiết bị; xử lý batch thực cho NT-Xent; giữ toàn bộ val/test và phần đuôi khi sinh nhãn giả; giữ BatchNorm/dropout encoder ở eval khi linear evaluation; truyền đúng tỷ lệ nhãn vào SupCon và tìm checkpoint cùng experiment/run/seed. Đợt kiểm chứng hiện tại mới xác nhận các nhánh này chạy thực tế trên hai bộ âm thanh.

## 2 Các thay đổi trong đợt này

| File hoặc nhóm file | Việc làm | Lý do |
|---|---|---|
| `.venv`, `scripts/install_deps_ipv4.py` | Cài PyTorch 2.5.1 CUDA 12.1; kiểm tra GPU ngoài sandbox; giữ các wheel tải hoàn chỉnh trong `.wheelhouse` | Khắc phục thiếu `torch`; tải IPv4 và tận dụng gói đã tải khi mạng timeout |
| `scripts/package_audio_data.py` | Đóng gói `.npz` đã xử lý thành `.pt` ở `runtime_data/smoke` hoặc thư mục đích người dùng chọn; sao chép manifest và normalization | Repo nạp tensor `.pt`; không xử lý lại WAV và không ghi đè workspace dữ liệu |
| `trainer/trainer.py` | Lưu `history.json`, loss hữu hạn và chỉ số train cân theo số mẫu | Theo dõi được quá trình chạy; tránh trung bình sai khi batch cuối nhỏ |
| `trainer/trainer.py` | Lưu `ckp_best.pt` theo validation loss cho supervised, linear evaluation và fine-tune; đánh giá test bằng checkpoint được chọn | Không chọn mô hình bằng test hoặc mặc định lấy epoch cuối |
| `trainer/trainer.py` | Lưu `resume.pt` mỗi epoch với model, temporal model, optimizer, scheduler và trạng thái RNG | Tiếp tục từ ranh giới epoch nếu bị gián đoạn; chưa tiếp tục giữa batch |
| `main.py` | Thêm `--resume`; chặn ghi đè run cũ và kiểm tra cấu hình tương thích; ưu tiên checkpoint tốt nhất ở các giai đoạn phụ thuộc | Tránh trộn kết quả và dùng sai checkpoint |
| `main.py` | Lưu cấu hình đã resolve và SHA-256 manifest trong `run_config.json` | Biết chính xác mô hình dùng tham số và dữ liệu nào |
| `scripts/run_audio_pipeline.py` | Lưu trạng thái từng giai đoạn, thời gian, lệnh chạy và mã thoát; hỗ trợ `--resume` để bỏ qua giai đoạn hoàn tất | Theo dõi và chạy lại có kiểm soát |
| `scripts/verify_audio_bundles.py` | Đối chiếu WAV thật với tensor test đã lưu và logits sau export, ít nhất một mẫu mỗi lớp | Chứng minh suy luận dùng cùng tiền xử lý với huấn luyện |
| `scripts/verify_audio_repo.py` | Một lệnh chạy đóng gói, toàn pipeline cả hai bộ, export, kiểm tra resume và parity; gom metrics, log, package versions và hash mã nguồn | Có bằng chứng tái lập cho báo cáo |
| `.gitignore` | Bỏ qua runtime data/jobs và wheel tải về | Tránh trộn dữ liệu lớn và tệp sinh ra với mã nguồn |

Self-supervised và SupCon chưa có tiêu chí validation contrastive để chọn epoch; hai giai đoạn này dùng checkpoint cuối. Các giai đoạn phân loại chọn epoch theo validation cross entropy. `ckp_last.pt` vẫn được giữ để đối chiếu.

## 3 Luồng kiểm chứng

```text
.npz đã xử lý
  → đóng gói .pt vào runtime_data/smoke
  → self_supervised
  → train_linear_10p
  → ft_10p
  → gen_pseudo_labels trên train
  → SupCon
  → train_linear_SupCon_10p
  → ft_SupCon_10p
  → supervised làm đối chứng 100% nhãn
  → export TorchScript
  → kiểm tra nạp lại resume
  → đối chiếu WAV và tensor test
```

Chạy riêng cho từng bộ, không gộp hai không gian nhãn. Supervised dùng toàn bộ nhãn của tập train nhỏ, nên là đối chứng khác ngân sách nhãn với fine-tune 10%; không coi là so sánh ngang ngân sách.

## 4 Lệnh tái lập

Từ thư mục repo:

```bash
.venv/bin/python scripts/verify_audio_repo.py --experiment smoke_audio_20261017 --device cuda:0
```

Nếu lượt này gián đoạn, dùng đúng cấu hình và thêm `--resume`. Muốn thí nghiệm mới thì dùng tên experiment khác. Với máy không có CUDA, dùng `--device cpu`.

Bộ nhỏ có Speech Commands train/val/test 700/700/700 và 70 nhãn thật ở mức 10%; RAVDESS 160/156/156 và 16 nhãn thật ở mức 10%. Các số lượng này đã đối chiếu với manifest và log chạy thực tế.

Để đóng gói dữ liệu đầy đủ mà không tiền xử lý lại:

```bash
.venv/bin/python scripts/package_audio_data.py --source data --output runtime_data/full
```

Sau đó mới chạy experiment riêng với nhiều epoch; không dùng kết quả bộ nhỏ để khẳng định chất lượng trên toàn bộ dữ liệu.

## 5 Bằng chứng và kết quả

**Đã hoàn thành kiểm chứng ngày 17/10/2026:** cả 16 bước của pipeline đạt (14 bước huấn luyện một epoch và 2 bước sinh nhãn giả); export, nạp lại resume và CLI dự đoán WAV đều đạt cho cả hai bộ. Đối chiếu frontend/logits đạt trên 35 WAV Speech Commands và 8 WAV RAVDESS, mỗi lớp một mẫu.

Fine-tune CA-TCC cuối của smoke test đạt Accuracy 4,43% và macro-F1 2,04% trên 700 mẫu Speech Commands; Accuracy 14,74% và macro-F1 5,32% trên 156 mẫu RAVDESS. Đây là số liệu kiểm tra chương trình một epoch, không phải chất lượng cuối cùng của phương pháp.


Trạng thái mới nhất nằm ở `artifacts/RUN_STATUS.md`. Đợt kiểm chứng lưu mọi lệnh và kết quả trong `artifacts/smoke_audio_20261017/verification_status.json`; log riêng từng bước trong cùng thư mục. Metrics từng giai đoạn nằm trong `experiments_logs/smoke_audio_20261017/`.

- `metrics_summary.csv` và `.json`: bảng số liệu tổng hợp, chỉ có sau khi toàn bộ kiểm chứng thành công.
- `wav_parity.json`: số mẫu đối chiếu, sai lệch đặc trưng/logits và nhãn dự đoán.
- `pipeline_status.json`: trạng thái từng giai đoạn và thời gian chạy.
- `history.json`: loss/accuracy theo epoch và thời gian epoch.
- `selection.json`: checkpoint được chọn và validation loss.
- `labels/*.npy`, báo cáo phân lớp `.xlsx` và confusion matrix `.torch`: kết quả đánh giá test.
- `packages.log`, `source_sha256.json`: phiên bản thư viện và mã nguồn tương ứng.
- `cli_inference_checks.json`, `predict_cli_*.log`: lệnh và kết quả thật của CLI suy luận WAV.
- `figures/*.png`: confusion matrix đã xuất và kiểm tra trực quan để chèn vào Word.

Để tạo lại bảng báo cáo và hình sau khi kiểm chứng thành công:

```bash
.venv/bin/python -m pip install -r requirements-report.txt
MPLCONFIGDIR=/tmp/ca_tcc_mpl .venv/bin/python scripts/summarize_audio_run.py --experiment smoke_audio_20261017
```

Workspace hiện không được Git nhận dạng là checkout hợp lệ nên chưa tạo commit. Hash SHA-256 của mã nguồn dùng trong lượt chạy được lưu để đối chiếu.

Các lần tải thất bại không được tính là hoàn thành cài đặt. Chỉ đánh dấu repo chạy được khi kiểm chứng runtime hoàn tất.

## 6 Phần ngoài phạm vi đã dừng

Một bản nháp web và bộ kiểm tra backend đã được tạo trước khi chốt lại ưu tiên. Server đã dừng theo yêu cầu; không đưa web vào tiêu chí nghiệm thu đợt này. Bảy kiểm tra backend dùng test double, không phải kết quả CA-TCC và không được đưa vào bảng Accuracy/F1 nghiên cứu.

## 7 Lượt huấn luyện đầy đủ theo yêu cầu tiếp theo

Đã khởi chạy experiment `full_audio_10p_seed0_20261017`, 40 epoch/giai đoạn, 10% nhãn, seed 0 trên toàn bộ train/val/test hai bộ; RAVDESS trước rồi Speech Commands.

- Đóng gói dữ liệu đầy đủ vào `runtime_data/full`, không chạy lại frontend.
- Thêm `scripts/run_full_audio.py` để kiểm tra môi trường, lưu snapshot mã nguồn và chạy pipeline trong dịch vụ nền của tài khoản.
- Thêm `scripts/report_full_audio.py` để xuất bảng chỉ số, CSV dự đoán theo file, confusion matrix và hình loss/accuracy sau khi mỗi bộ hoàn tất.
- Sau mỗi bộ, tự export checkpoint CA-TCC tốt nhất; sau cả hai bộ, tự kiểm tra WAV parity.
- Dịch vụ: `ca-tcc-full-20261017-10p.service`. Trạng thái thực tế luôn nằm ở `artifacts/full_audio_10p_seed0_20261017/full_run_status.json`; `training` chưa phải hoàn tất.
- Kết quả chạy thử trước đó được giữ riêng; không trộn vào bảng đánh giá đầy đủ.

## Bổ sung ngày 17/10/2026: lượt 1% nhãn thật

Khởi chạy riêng `full_audio_1p_seed0_20261017` cho RAVDESS và Speech Commands, 40 epoch/pha, seed 0. Dùng 10/960 và 848/84.843 mẫu có nhãn tương ứng; giữ đủ 8 và 35 lớp. Thêm tùy chọn thư mục dữ liệu riêng vào supervisor để bảo toàn nhãn giả của lượt 10%, và bỏ chạy lại baseline supervised 100% đã có. Chi tiết: `artifacts/full_audio_1p_seed0_20261017/THONG_TIN_LUOT_CHAY.md`.

## Hoàn tất 1% nhãn và so sánh 10%

Lượt 1% hoàn tất lúc 15:33:34 ngày 17/10/2026. Đối chiếu các pha hoàn thành, lịch sử đủ 40 epoch, checkpoint, nhãn test, Accuracy/macro-F1 từ dự đoán và parity 43 WAV đều đạt. Đã lưu `SO_SANH_1P_10P.md/csv/json/png`, script dựng bảng/hình và `completion_verification.json` trong artifacts của lượt 1%. Attempt self-supervised Speech Commands bị ngắt trước resume được đánh dấu interrupted sau khi toàn lượt hoàn thành; bản trạng thái trước chú thích đã được giữ. DINOv3 chưa chạy.
