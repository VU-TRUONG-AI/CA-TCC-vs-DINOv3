# Web phân tích bộ âm thanh CA-TCC

Ứng dụng chạy trên máy cá nhân. Mỗi lượt xử lý một tác vụ: Speech Commands (35 từ tiếng Anh) hoặc RAVDESS (8 cảm xúc). Hỗ trợ ZIP nhiều WAV, kiểm tra trước khi chạy, CSV nhãn tùy chọn, tiến độ xử lý nền, lịch sử, nghe lại, lọc kết quả, phân bố dự đoán, confusion matrix và xuất báo cáo.

## Khởi động

```bash
.venv/bin/python -m web.server --port 8000
```

Mở http://127.0.0.1:8000. Backend chỉ lắng nghe localhost; không đưa trực tiếp lên mạng công cộng. Dữ liệu và kết quả lưu ở `runtime_jobs/`. Khi khởi động lại, tác vụ chưa hoàn tất được ghi nhận thất bại; không tự chạy tiếp.

## Chuẩn bị ZIP và nhãn

```text
batch.zip
├── folder/audio_001.wav
├── folder/audio_002.wav
└── labels.csv                 # tùy chọn
```

```csv
file,label
folder/audio_001.wav,happy
folder/audio_002.wav,sad
```

Hoặc chọn tệp nhãn riêng khi tải ZIP. Dùng tên nhãn tiếng Anh đúng bộ lớp. CSV chỉ cần chứa những file có nhãn; không tính metrics trên file thiếu nhãn hoặc file lỗi. Đường dẫn phải khớp chính xác và không được trùng. Không suy nhãn tự động từ tên file/thư mục.

Giới hạn: ZIP 512 MiB, 50.000 mục, mỗi file 32 MiB, tổng sau giải nén 2 GiB. Chỉ đọc WAV và `labels.csv` ở gốc ZIP; file khác được bỏ qua. File quá 1 giây (Speech Commands) hoặc 6 giây (RAVDESS) được báo lỗi để chia đoạn trước; không âm thầm cắt. WAV ngắn được đệm theo frontend gốc.

## Bundle mô hình

```text
artifacts/models/SpeechCommands/{classifier.ts,metadata.json,normalization.npz}
artifacts/models/RAVDESS/{classifier.ts,metadata.json,normalization.npz}
```

Mô hình được tải một lần cho mỗi tác vụ, dùng CPU để dành GPU cho huấn luyện. Các lượt dự đoán chạy tuần tự, batch 32. Nếu chưa có bundle, vẫn có thể kiểm tra bộ dữ liệu nhưng chưa chạy dự đoán. Bundle smoke được hiển thị rõ là chỉ kiểm tra chương trình. Khởi động lại server sau khi đổi bundle.

## Kết quả

- `predictions.csv`: dự đoán và điểm softmax của từng file thành công, nhãn thật và đúng/sai nếu có.
- `errors.csv`: tên file và lý do lỗi.
- `metrics.json`: chỉ tạo khi có ít nhất một file dự đoán thành công có nhãn thật.
- `report.html`: báo cáo độc lập, mở bằng trình duyệt.

Macro-F1 lấy trung bình trên toàn bộ lớp của mô hình; lớp không xuất hiện đóng góp 0. Balanced accuracy tính trung bình recall trên các lớp có nhãn thật trong lượt đánh giá. Confusion matrix có hàng là nhãn thật, cột là dự đoán. Thống kê dự đoán không chứng minh chất lượng khi thiếu nhãn thật. Điểm softmax chưa được calibration.

## API

| Phương thức | Đường dẫn | Chức năng |
|---|---|---|
| GET | `/api/models` | Trạng thái bundle và các lớp |
| POST | `/api/jobs?dataset=RAVDESS` | Body là ZIP nhị phân; kiểm tra và tạo lượt |
| POST | `/api/jobs/{id}/labels` | Body là CSV UTF-8, trước khi bắt đầu |
| POST | `/api/jobs/{id}/start` | Xếp hàng chạy |
| GET | `/api/jobs` | Lịch sử |
| GET | `/api/jobs/{id}` | Trạng thái và kết quả |
| GET | `/api/jobs/{id}/download/{filename}` | Tải báo cáo |
| GET | `/api/jobs/{id}/audio/{path}` | Nghe WAV |

## Dữ liệu huấn luyện

Đóng gói đặc trưng đã lưu sang thư mục riêng, không chạy lại tiền xử lý:

```bash
.venv/bin/python scripts/package_audio_data.py --source datasets/processed/smoke --output datasets/tensors/smoke
.venv/bin/python scripts/run_audio_pipeline.py --data-root datasets/tensors/smoke --experiment smoke_audio_20261017 --epochs 1 --label-percent 10 --device auto
```

Huấn luyện supervised/fine-tune lưu `ckp_best.pt` theo validation loss; bước kế tiếp ưu tiên checkpoint này. Self-supervised/SupCon dùng checkpoint cuối. `history.json` lưu loss/accuracy theo epoch, `selection.json` ghi epoch được chọn. Smoke không thay thế thực nghiệm nhiều epoch và nhiều seed trên dữ liệu đầy đủ.
