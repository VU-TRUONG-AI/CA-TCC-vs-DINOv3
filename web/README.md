# DINOv3 Audio API và app demo

Hai mô hình riêng: **RAVDESS** (8 cảm xúc, cửa sổ WAV tối đa 6 giây) và **SpeechCommands** (35 nhãn, tối đa 1 giây). Cả hai dùng encoder DINOv3 ViT-S/16, model suy luận chạy CPU, không tranh GPU huấn luyện.

## Chạy app/API

Từ thư mục gốc repo:

```bash
.venv-dinov3/bin/python -m web.server --port 8000 --bundles artifacts/dinov3_app_bundles --jobs runtime_jobs/dinov3
```

Mở http://127.0.0.1:8000. `GET /api/models` hiển thị `ready` và `backbone` của từng tác vụ. Một model chỉ sẵn sàng sau khi gói xuất của nó hoàn tất kiểm tra. Không trỏ tới bundle CNN khi demo DINOv3.

## Dự đoán một WAV từ app khác

```bash
curl --fail -X POST 'http://127.0.0.1:8000/api/predict?dataset=RAVDESS' \
  -H 'Content-Type: audio/wav' --data-binary @/absolute/path/emotion.wav

curl --fail -X POST 'http://127.0.0.1:8000/api/predict?dataset=SpeechCommands' \
  -H 'Content-Type: audio/wav' --data-binary @/absolute/path/command.wav
```

Body là byte WAV, không phải multipart. Kết quả JSON có `dataset`, `backbone`, `version`, `duration` và `top_k` (label/probability). API cache model sau lần nạp đầu. Điểm softmax không phải xác suất đã hiệu chỉnh.

App cũng hỗ trợ ZIP nhiều WAV qua `POST /api/jobs?dataset=...`, `POST /api/jobs/<id>/start`, `GET /api/jobs/<id>` và tải CSV/báo cáo khi xong. ZIP có nhãn thật mới tính accuracy/F1; không tạo chỉ số chất lượng cho ZIP không nhãn.

## Nhúng model trực tiếp trong backend Python

Mỗi thư mục trong `artifacts/dinov3_app_bundles/<dataset>/` có `classifier.ts`, `metadata.json`, `normalization.npz`, frontend, script inference và cấu hình huấn luyện. Copy cả thư mục, cài `requirements.txt`, chạy `python infer.py --wav /absolute/path.wav`.

`classifier.ts` nhận log-mel đã chuẩn hóa `[B,40,T]`, trả `(logits, temporal_features)`. Image adapter DINOv3 đã nằm trong file. Không cần checkout DINOv3/trọng số pretrained trong app phục vụ suy luận. `verification.json` ghi so khớp logits với checkpoint PyTorch và so khớp WAV thật. Gói này dùng Python/PyTorch ở backend; chuyển sang native Android/iOS là mục tiêu định dạng khác.

Server mặc định bind localhost và chặn Origin ngoài localhost. Để nhúng vào backend hiện có, dùng bundle trực tiếp hoặc proxy cùng origin. Hạn nộp: 19:00 ngày 18/10/2026; báo cáo huấn luyện ghi rõ ngân sách rút gọn 4 epoch của SpeechCommands.

### Chuyển đổi âm thanh miễn phí

Giao diện nhận WAV và các định dạng mà trình duyệt giải mã được (MP3, M4A/AAC, FLAC, OGG/Opus, WebM). File nén được giải mã bằng Web Audio, trộn mono và chuyển thành WAV PCM16 16 kHz trên thiết bị. Không cần dịch vụ chuyển đổi trả phí. Nút **Tải bản WAV miễn phí** lưu WAV; chính bản WAV chuyển đổi được dùng khi gọi API hoặc suy luận trong trình duyệt. WAV đầu vào giữ đường xử lý hiện có.

Giới hạn file 32 MiB; chuyển đổi tối đa 10 phút, còn suy luận vẫn giới hạn 6 giây cho RAVDESS và 1 giây cho SpeechCommands. Codec hỗ trợ phụ thuộc trình duyệt; file lỗi hoặc codec không hỗ trợ được báo trên giao diện. Kiểm tra `scripts/qa_audio_conversion.mjs` tạo bản ghi WebM/Opus, xác nhận WAV mono/16 kHz/PCM16, tải file, chạy suy luận thật và kiểm tra file hỏng.
