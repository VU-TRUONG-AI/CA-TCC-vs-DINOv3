# Sắp xếp dự án ngày 17/10/2026

## Kết quả

Trong thư mục `CĐ` chỉ còn một dự án: `CA-TCC-main`. Toàn bộ mã, dữ liệu và kết quả nằm bên trong. Mở `README.md` ở thư mục gốc để tìm đúng nơi làm việc.

## Đã chuyển những gì?

| Trước | Sau | Lý do |
|---|---|---|
| `../CA-TCC-data-workspace/` | `datasets/` | Gom dữ liệu và tiền xử lý vào cùng dự án |
| `runtime_data/` | `datasets/tensors/` | Đặt NPZ và PT trong cùng nhóm dữ liệu |
| `Baibao/` | `documents/Baibao/` | Gom tài liệu nghiên cứu |
| `Slide_word/` | `documents/Slide_word/` | Gom tài liệu trình bày |
| `../CA-TCC-archive/` | `docs/reference/legacy_20261013.tar.gz` | Nén mã không dùng, vẫn có thể khôi phục |

Mã mô hình giữ cấu trúc Python hiện có (`models`, `trainer`, `dataloader`, `config_files`, `scripts`) để import và lệnh chạy tiếp tục ổn định. Kết quả huấn luyện giữ tại `artifacts` và `experiments_logs`.

## Đã xóa những gì?

- Cache bánh xe cài đặt `.wheelhouse/`: 1.483.726.321 byte, khoảng 1,48 GB. Thư viện đã cài trong `.venv/` vẫn được giữ.
- Cache Python `__pycache__` có thể tự sinh lại.
- Cấu hình VSCode và liên kết `.venv` lặp trong thư mục dữ liệu; dùng cấu hình/môi trường ở gốc.
- Thư mục rỗng `runtime_jobs/`; ứng dụng web tự tạo lại khi cần.
- Hai thư mục CA-TCC bên cạnh được bỏ sau khi di chuyển dữ liệu và kiểm tra gói nén mã cũ.

Giữ toàn bộ WAV, ZIP dữ liệu gốc, NPZ, PT, normalization, nhãn giả, checkpoint, lịch sử huấn luyện, kết quả smoke/full, bài báo và hai bản slide. Không huấn luyện lại hoặc thay đổi kết quả đánh giá trong bước sắp xếp này.

## Đường dẫn ban đầu sau lần gom thứ nhất (đã thay thế bên dưới)

- `Data → datasets/raw`
- `data → datasets/processed/full`
- `data_smoke → datasets/processed/smoke`
- `runtime_data → datasets/tensors`

Các liên kết trên đã được bỏ ở lần làm gọn tiếp theo, xem mục cuối. `scripts/audio_features.py` đã trỏ tới frontend chuẩn ở `datasets/pipeline/`.

`datasets/pipeline/package_tensors.py` dùng chung hàm đóng gói với `scripts/package_audio_data.py` và xuất PT vào `datasets/tensors/`, tránh tạo thêm PT lẫn trong `processed/`.

Manifest dữ liệu, danh mục audit và tài liệu đang dùng được cập nhật vị trí WAV mới. Danh sách split, thứ tự mẫu, nhãn và số lượng giữ nguyên. Các bản trước khi đổi đường dẫn nằm trong `reorganization_20261017/path_documents_before.zip`; hash trước/sau và danh sách thao tác nằm trong `migration.json` cùng thư mục.

Log huấn luyện, `run_config.json`, snapshot mã nguồn và báo cáo lịch sử vẫn ghi đường dẫn tại thời điểm chạy. Không sửa các bằng chứng này. Khi đọc đường dẫn cũ, thay `/home/truongvu/Documents/CĐ/CA-TCC-data-workspace` bằng `/home/truongvu/Documents/CĐ/CA-TCC-main/datasets`. Hash manifest trong log cũ khớp bản trước di chuyển, không khớp bản đã cập nhật đường dẫn; không dùng chênh lệch hash này để suy ra dữ liệu train đã đổi.

## Kiểm chứng

Nhật ký kiểm tra sau sắp xếp: `reorganization_20261017/verification.json` và các tệp log cùng thư mục. Kiểm tra đường dẫn WAV, split/nhãn của manifest trước/sau, đọc tensor, import frontend và suy luận WAV so với tensor qua hai mô hình đã xuất. Đối chiếu inode/kích thước/mtime của 107.393 tệp ngay sau đổi vị trí cho thấy thao tác di chuyển giữ nguyên các tệp.

Kết quả kiểm tra thực tế: **đạt tất cả**. Đối chiếu 8 manifest chỉ thay đổi đường dẫn; 107.320 tệp WAV/NPZ/ZIP giữ nguyên inode, kích thước và mtime. Kiểm tra dữ liệu đầy đủ của cả hai bộ đạt, gồm normalization, nhãn, shape, hữu hạn và tách người nói. Kiểm tra 43 WAV qua hai mô hình đã xuất đạt. Lệnh đóng gói dùng chung chạy thành công cho full và smoke; lệnh CLI huấn luyện nạp thành công.

## Làm gọn tiếp: chỉ còn một mục dữ liệu ở gốc

Theo yêu cầu người dùng, đã xóa bốn liên kết `Data`, `data`, `data_smoke`, `runtime_data`, giữ nguyên thư mục đích `datasets/`. Không xóa dữ liệu. Các lệnh đóng gói, huấn luyện, export, báo cáo smoke và phần đọc nhãn của bản nháp web chuyển sang `datasets/processed/` hoặc `datasets/tensors/`. Defaults của main, pipeline và export đều dùng PT đầy đủ tại `datasets/tensors/full`.

Hình phương pháp trong `misc/` được chuyển vào `docs/reference/figures/` và sửa liên kết trong README upstream. Giữ cấu trúc mã Python hiện tại để tránh thay đổi import của quá trình đang huấn luyện.

Snapshot mã trước thay đổi đường dẫn: `reorganization_20261017/source_before_remove_aliases.zip`. Chi tiết thay đổi: `remove_aliases.json` cùng thư mục. Các snapshot và log huấn luyện cũ vẫn được giữ nguyên; đường dẫn lịch sử `runtime_data/` nay tương ứng `datasets/tensors/`, `data/` tương ứng `datasets/processed/full/`, `data_smoke/` tương ứng `datasets/processed/smoke/`, `Data/` tương ứng `datasets/raw/`. Khi chạy lại lệnh trong báo cáo cũ, đổi sang đường dẫn hiện hành. Script báo cáo đầy đủ hỗ trợ tự ánh xạ `runtime_data` trong cấu hình lịch sử.

Lượt 1% đang chạy dùng trực tiếp `datasets/tensors/experiments/full_audio_1p_seed0_20261017`, không phụ thuộc bốn liên kết vừa bỏ. Thay đổi lần này chỉ liên quan vị trí tệp và đường dẫn mặc định, không thay đổi kiến trúc, loss, optimizer, augmentation hoặc siêu tham số huấn luyện.

Kiểm tra sau khi bỏ liên kết: export bằng đường dẫn mặc định mới đạt; suy luận 43 WAV qua cả hai mô hình khớp tensor; tiến trình 1% vẫn cập nhật trạng thái và lịch sử epoch. Chi tiết: `reorganization_20261017/remove_aliases_verification.json`.
