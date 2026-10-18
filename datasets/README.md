# Dữ liệu và tiền xử lý

Đây là phần dữ liệu của cùng dự án CA-TCC. Chỉ cần mở thư mục cha `CA-TCC-main`, dùng môi trường `.venv` ở thư mục gốc.

| Thư mục | Nội dung |
|---|---|
| `raw/` | WAV gốc Speech Commands, RAVDESS và gói ZIP gốc |
| `processed/full/` | Đặc trưng NPZ đầy đủ, normalization và manifest chia train/val/test |
| `processed/smoke/` | Bộ nhỏ để kiểm tra code |
| `tensors/full/` | Tensor PT dùng huấn luyện đầy đủ và nhãn giả |
| `tensors/smoke/` | Tensor PT dùng kiểm tra code |
| `pipeline/` | Mã tiền xử lý, kiểm tra dữ liệu và dựng hình |
| `results/` | Audit, kiểm tra từng bước, kiểm chứng dữ liệu |
| `reports/` | Báo cáo tiếng Việt để tham khảo khi viết Word |
| `figures/` | Hình PNG/SVG cho báo cáo và slide |
| `logs/` | Nhật ký chạy phần dữ liệu |
| `docs/` | Mô tả đầu ra, các bước xử lý và lịch sử dọn dẹp |
| `environment/` | Thông tin môi trường và requirements phần dữ liệu |

Mỗi bộ dữ liệu có nhóm riêng `SpeechCommands` và `RAVDESS`; không gộp hai không gian nhãn.

Từ **thư mục gốc dự án**:

```bash
.venv/bin/python datasets/pipeline/run_pipeline.py --step validate
.venv/bin/python datasets/pipeline/run_pipeline.py --step figures
```

Không cần chạy lại tiền xử lý để dùng dữ liệu đã có. Toàn bộ 107.269 WAV đã được xử lý. Huấn luyện đầy đủ ở mức 10% nhãn thật, 40 epoch/giai đoạn, seed 0 đã hoàn thành cho hai bộ; xem [báo cáo huấn luyện](../artifacts/full_audio_10p_seed0_20261017/KET_QUA_HUAN_LUYEN.md).

Đọc [báo cáo tiền xử lý](reports/BAO_CAO_TIEN_XU_LY.md) và lấy hình trong `figures/<dataset>/`. Checkpoint nằm ngoài phần dữ liệu, tại `../experiments_logs/`; kết quả mô hình và export tại `../artifacts/`.

Ngày 17/10/2026, phần này được di chuyển từ workspace bên cạnh vào dự án; chỉ cập nhật đường dẫn trong danh mục dữ liệu. Không thay đổi WAV, NPZ, tensor, split, nhãn hay normalization. Các log huấn luyện cũ và snapshot được giữ nguyên để truy vết; đường dẫn lịch sử được giải thích ở [nhật ký sắp xếp](../docs/SAP_XEP_THU_MUC.md).
