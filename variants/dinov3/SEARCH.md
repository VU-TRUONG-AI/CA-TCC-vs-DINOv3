# Tìm tham số DINOv3 với 10% nhãn

## Trạng thái ngày 17/10/2026

**Cập nhật sau khi nhận email cấp quyền:** Đã tải `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, xác minh SHA-256 `08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d`, nạp strict thành công. Kiểm tra GPU pretrained batch 8 qua cả hai dataset (`artifacts/dinov3_preflight_20261017/pretrained_probe.json`). Đã bắt đầu search, trial đầu tiên RAVDESS `t00`, lr=1e-5, weight decay=1e-4, batch 8. Trạng thái trực tiếp: `artifacts/dinov3_search_10p_20261017/search_status.json`. Chưa có cấu hình thắng khi bắt đầu lượt đầu.

Các ghi chú “thiếu weights/chưa khởi chạy” dưới đây là lịch sử trước khi nhận email, được giữ để truy vết.

- Baseline 1% đã hoàn tất, gồm export và WAV parity.
- Đã tải mã nguồn chính thức vào `external/dinov3`; không sửa mã upstream.
- Môi trường `.venv-dinov3` có torchvision 0.20.1+cu121, pandas 2.2.3 và openpyxl 3.1.5 riêng, dùng lại thư viện baseline qua đường dẫn chỉ đọc và system site packages. Đây không phải môi trường hoàn toàn độc lập; không sửa/cập nhật thư viện `.venv` baseline. Entry point `main.py --help` đã chạy thành công. Danh sách package được lưu tại `artifacts/dinov3_preflight_20261017/packages.log`.
- Đã chạy kiểm tra kỹ thuật GPU bằng dữ liệu giả và khởi tạo ngẫu nhiên cho RAVDESS và SpeechCommands. Các pha self_supervised, SupCon, ft_10p và train_linear_10p đều chạy forward/backward ở batch 2 và 8. RAVDESS batch 32 hết VRAM, SpeechCommands batch 32 chạy được. Chi tiết: `artifacts/dinov3_preflight_20261017/probe.json`.
- Đã kiểm tra bộ điều phối bằng test tự động: thiếu weights không khởi chạy, lưu trial lỗi, chọn theo validation loss, resume không lặp trial hoàn tất, trainer không đánh giá test khi tìm tham số.
- **Chưa chạy huấn luyện pretrained hoặc tìm được tham số tốt nhất. Thiếu checkpoint ViT-S/16 chính thức.** Đã gọi search và lưu trạng thái `awaiting_weights` tại `artifacts/dinov3_search_10p_20261017/search_status.json`.

Meta yêu cầu đăng ký quyền truy cập weights tại https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/. Đặt checkpoint ViT-S/16 LVD-1689M được cấp vào `weights/dinov3_vits16.pth`. Không dùng ViT-S+/B/L hoặc tự thay bằng DINOv2. Không tự chuyển sang random nếu thiếu weights.

## Ngân sách và tiêu chí

Mỗi bộ dữ liệu thử 6 tổ hợp:

| Tham số | Giá trị |
|---|---|
| learning rate (Adam của model và TC) | 1e-5, 3e-5, 1e-4 |
| weight decay (model và TC) | 1e-4, 3e-4 |
| batch size | 8 |
| seed | 0 |
| nhãn thật | 10% train |
| epoch mỗi pha học trong trial | 10 |
| epoch mỗi pha học trong lượt cuối | 40 |
| backbone | DINOv3 ViT-S/16 pretrained, trainable |

Chạy tuần tự 12 trial tổng cộng, sau đó một lượt đủ 40 epoch cho cấu hình thắng của mỗi dataset (14 lượt pipeline tổng cộng). Mỗi pipeline giữ các pha CA-TCC: self_supervised → linear → fine-tune → pseudo labels → SupCon → linear SupCon → fine-tune SupCon. Self-supervised và SupCon vẫn dùng toàn bộ train; nhãn thật 10% dùng ở các pha phân loại. Các lượt cuối khởi tạo lại từ pretrained, không kéo dài trial 10 epoch.

Checkpoint trong mỗi pha phân loại được chọn theo validation cross entropy nhỏ nhất. Xếp hạng trial theo validation loss của checkpoint đã chọn ở pha `ft_SupCon_10p`. Không dùng accuracy test để chọn hoặc điều chỉnh tham số. Chỉ mô hình cuối được đánh giá test, báo accuracy, macro-F1, balanced accuracy, confusion matrix và CSV dự đoán. Chỉ có một seed; đây là cấu hình tốt nhất trong tập thử với ngân sách này, không phải tối ưu toàn cục. Xếp hạng ở 10 epoch có thể khác 40 epoch.

Batch 8 khác batch 32 của baseline CNN: số negatives và số bước optimizer thay đổi, vì vậy không mô tả so sánh là chỉ thay encoder. Chưa đo thời gian hoàn thành toàn bộ tập dữ liệu.

## Chạy hoặc tiếp tục sau khi có checkpoint

Từ thư mục gốc dự án, dùng GPU:

```bash
.venv-dinov3/bin/python variants/dinov3/search.py --resume
```

Search mặc định dùng tên `dinov3_search_10p_20261017`, đã lưu cấu hình và đang chờ weights. Các lệnh trong môi trường sandbox cần được cho phép truy cập GPU. Thay đổi grid, batch, epoch, seed hoặc nguồn weights sau khi bắt đầu phải dùng tên experiment mới. Không chạy hai tiến trình cho cùng search; file lock ngăn trùng lặp.

Không có tiến trình tự chờ weights. Search hiện chạy trong phiên terminal duy trì, thông tin tại `artifacts/dinov3_search_10p_20261017/runtime_session.json`. Không gọi thêm search khi phiên này còn hoạt động. `--resume` bỏ qua trial hoàn tất, thử lại trial lỗi và giữ lịch sử attempt trước đó. Tiếp tục checkpoint của pipeline nếu có. Source/checkpoint hash thay đổi sẽ bị chặn. Nếu tất cả trial của một dataset thất bại, search dừng để xử lý lỗi, không báo cấu hình tốt nhất.

## Kết quả lưu ở đâu

- `artifacts/<search>/search_status.json`: cấu hình tìm kiếm, trạng thái, command, giờ bắt đầu/kết thúc, mã thoát, lỗi, từng trial, finalists và test cuối.
- `artifacts/<search>/trials.csv`: bảng tham số + validation loss/accuracy + epoch được chọn cho mọi trial, kể cả lỗi.
- `artifacts/<search>/<trial>.log`: log điều phối riêng, append khi resume.
- `artifacts/<search>/best_parameters_<dataset>.json`: cấu hình thắng theo validation.
- `artifacts/<trial>/`: pipeline log, package versions và source snapshot/hash.
- `experiments_logs/<trial>/<dataset>/<stage>_seed_0/`: resolved config, history, selection, best/last/resume checkpoints.
- Trong pha cuối của finalist: `winner_test_metrics.json` và `winner_test_predictions.csv`.
- `datasets/tensors/experiments/<trial>/`: liên kết dữ liệu đầu vào, manifest/normalization riêng, pseudo labels riêng.

Không ghi đè kết quả CNN. Việc xuất TorchScript DINOv3 chưa được triển khai.
